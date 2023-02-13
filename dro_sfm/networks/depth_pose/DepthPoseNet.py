from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from dro_sfm.geometry.pose_utils import pose_mat2vec

from dro_sfm.networks.optim.update import BasicUpdateBlockPose, BasicUpdateBlockDepth
from dro_sfm.networks.optim.update import DepthHead, PoseHead, UpMaskNet
from dro_sfm.networks.optim.extractor import ResNetEncoder

from dro_sfm.geometry.camera import Camera, Pose
from dro_sfm.utils.depth import inv2depth
from dro_sfm.networks.layers.resnet.layers import disp_to_depth
from dro_sfm.networks.model_depth_pose import Model_depth_pose
from dro_sfm.networks.layers.feature_pyramid import FeaturePyramid
from dro_sfm.networks.gmflow.backbone import CNNEncoder

                
class DepthPoseNet(nn.Module):
    def __init__(self, cfg, version=None, min_depth=0.1, max_depth=100, **kwargs):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        assert "it" in version
        self.iters = int(version.split("-")[0].split("it")[1])
        self.is_high = "h" in version
        self.out_normalize = "out" in version
        # get seq len in one stage. default: 4.
        self.seq_len = 4 
        for str in version.split("-"):
            if "seq" in str:
                self.seq_len = int(str.split("seq")[1])
        # update iters
        self.iters = self.iters // self.seq_len
        # intermediate supervision
        self.inter_sup =  "inter" in version

        print(f"=======iters:{self.iters}, sub_seq_len:{self.seq_len}, inter_sup: {self.inter_sup}, is_high:{self.is_high}, out_norm:{self.out_normalize}, max_depth:{self.max_depth} min_depth:{self.min_depth}========")
        
        if self.out_normalize:
            self.scale_inv_depth = partial(disp_to_depth, min_depth=self.min_depth, max_depth=self.max_depth)
        else:
            self.scale_inv_depth = lambda x: (x, None) # identity
        
        # feature network, context network, and update block
        self.foutput_dim = 128
        self.feat_ratio = 8
        # TODO
        # 输出维度作为参数传入， 网络返回结果有该输出维度的特征
        # self.fnet = FeaturePyramid()
        self.fnet = CNNEncoder(output_dim= self.foutput_dim)


        self.depth_pose_head = Model_depth_pose(cfg)

        self.upmask_net = UpMaskNet(hidden_dim=self.foutput_dim, ratio=self.feat_ratio)
        
        self.hdim = 128 
        self.cdim = 32
        
        self.update_block_depth = BasicUpdateBlockDepth(hidden_dim=self.hdim, cost_dim=self.foutput_dim, ratio=self.feat_ratio, context_dim=self.cdim)
        self.update_block_pose = BasicUpdateBlockPose(hidden_dim=self.hdim, cost_dim=self.foutput_dim, context_dim=self.cdim)

        self.cnet_depth = CNNEncoder(output_dim = self.hdim+self.cdim)
        self.cnet_pose = CNNEncoder(output_dim = self.hdim+self.cdim, input_images = 2)

        for param in self.fnet.parameters():
                param.requires_grad = False
        for param in self.depth_pose_head.parameters():
                param.requires_grad = False




    def upsample_depth(self, depth, mask, ratio=8):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, ratio*H, ratio*W)
    
    def get_cost_each(self, pose, fmap, fmap_ref, depth, K, ref_K, scale_factor):
        """
            pose: (b, 3, 4)
            depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        pose = Pose.from_vec(pose, "euler")

        # identity = torch.eye(4, device=pose.device, dtype=pose.dtype).repeat([len(pose), 1, 1])
        # identity[:,:3,:] = pose
        # pose = Pose(identity)

        device = depth.device
        cam = Camera(K=K.float()).scaled(scale_factor).to(device) # tcw = Identity
        ref_cam = Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device)
        
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)

        fmap_warped = F.grid_sample(fmap_ref, ref_coords, mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        cost = (fmap - fmap_warped)**2
        
        return cost
    
    def depth_cost_calc(self, inv_depth, fmap, fmaps_ref, pose_list, K, ref_K, scale_factor):
        cost_list = []
        for pose, fmap_r in zip(pose_list, fmaps_ref):
            cost = self.get_cost_each(pose, fmap, fmap_r, inv2depth(inv_depth), K, ref_K, scale_factor)
            cost_list.append(cost)  # (b, c,h, w)
        # cost = torch.stack(cost_list, dim=1).min(dim=1)[0]
        cost = torch.stack(cost_list, dim=1).mean(dim=1)
        return cost

    def points2depthmap(self, H, W, coord, depth, ratio):
        # 稀疏点深度恢复对应深度图
        # coord [b, n, 2] 深度点对应坐标
        # depth [b, n ,1] 三角化结果，所有深度点
        # ratio 4 or 8, 缩小的比率
        depth_map = torch.zeros(depth.shape[0], 1, int(H/ratio), int(W/ratio), device=depth.device)
        for batch in range(depth.shape[0]):
            for n in range(depth.shape[1]):
                h, w = torch.clamp((coord[batch, n, :] / ratio)[:], 0, 39 )
                depth_map[batch, 0, int(h), int(w)] = depth[batch, n, :]

        return depth_map        

            
    def forward(self, target_image, ref_imgs, intrinsics):
        """ Estimate inv depth and  poses """
        # run the feature network
        K, K_inv = [], []
        for intr in intrinsics.cpu().numpy():
            K.append(intr)
            K_inv.append(np.linalg.inv(intr))
        K = torch.tensor(K).cuda().unsqueeze(1)
        K= K.float()
        K_inv = torch.tensor(K_inv).cuda().unsqueeze(1)
        K_inv = K_inv.float()

        fmaps_target = self.fnet(target_image)  # fmaps 是6个尺度的特征列表
        fmaps_ref = []
        for img in ref_imgs:
            fmaps_ref.append(self.fnet(img))
        # assert target_image.shape[2] / fmap1.shape[2] == self.feat_ratio


        # 初始化深度和位姿
        pose_list_init = []
        for ref_img, fmap_ref in zip(ref_imgs, fmaps_ref):
            _, coord1, depth1, pose = self.depth_pose_head([target_image, ref_img, K, K_inv], [fmaps_target, fmap_ref])
            vec = pose_mat2vec(pose)
            pose_list_init.append(vec)

        # 三角变换出来的稀疏深度 inv_depth_init -> [b, 1, H/self.feat_ratio, W/self.feat_ratio]
        inv_depth_init  = self.points2depthmap(target_image.shape[2], target_image.shape[3], coord1, depth1, self.feat_ratio)

        # 深度图恢复原本W， H
        up_mask = self.upmask_net(fmaps_target[0])
        inv_depth_up_init = self.upsample_depth(inv_depth_init, up_mask, ratio=self.feat_ratio)

        # 预测结果 list
        inv_depth_predictions = [self.scale_inv_depth(inv_depth_up_init)[0]]
        pose_predictions = [[pose.clone() for pose in pose_list_init]]
        
        
        # run the context network for optimization
        if self.iters > 0:
            # 提取深度的隐藏特征和上下文特征
            cnet_depth = self.cnet_depth(target_image)[0]        
            hidden_d, inp_d = torch.split(cnet_depth, [self.hdim, self.cdim], dim=1)
            hidden_d = torch.tanh(hidden_d)
            inp_d = torch.relu(inp_d)
            
            # 提取位姿的隐藏特征和上下文特征
            cnet_pose_list = []
            for ref_img in ref_imgs:
                cnet_pose_list.append( self.cnet_pose(torch.cat([target_image, ref_img], dim=1))[0])  
            hidden_p_list, inp_p_list = [], []
            for cnet_pose in cnet_pose_list:
                hidden_p, inp_p = torch.split(cnet_pose, [self.hdim, self.cdim], dim=1)
                hidden_p_list.append(torch.tanh(hidden_p))
                inp_p_list.append(torch.relu(inp_p))
            
                
        # optimization start.................
        # 特征提取金字塔网络提取的多尺度特征，提取输出维度 128特征

        pose_list = pose_list_init
        inv_depth = inv_depth_init
        inv_depth_up = None
        for itr in range(self.iters):
            inv_depth = inv_depth.detach()
            pose_list = [pose.detach() for pose in pose_list]

            # calc cost
            pose_cost_func_list = []
            for fmap_ref in fmaps_ref:
                pose_cost_func_list.append(partial(self.get_cost_each, fmap=fmaps_target[0], fmap_ref=fmap_ref[0],
                                                   depth=inv2depth(self.scale_inv_depth(inv_depth)[0]),
                                                   K=intrinsics, ref_K=intrinsics, scale_factor=1.0/self.feat_ratio))

            depth_cost_func = partial(self.depth_cost_calc, fmap=fmaps_target[0], fmaps_ref=fmaps_ref[0],
                                      pose_list=pose_list, K=intrinsics,
                                      ref_K=intrinsics, scale_factor=1.0/self.feat_ratio)

            #########  update depth ##########
            hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block_depth(hidden_d, depth_cost_func,
                                                                             inv_depth, inp_d,
                                                                             seq_len=self.seq_len, 
                                                                             scale_func=self.scale_inv_depth)
            
            if not self.inter_sup:
                up_mask_seqs, inv_depth_seqs = [up_mask_seqs[-1]], [inv_depth_seqs[-1]]
            # upsample predictions
            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                inv_depth_up = self.upsample_depth(inv_depth_i, up_mask_i, ratio=self.feat_ratio)
                inv_depth_predictions.append(self.scale_inv_depth(inv_depth_up)[0])
            inv_depth = inv_depth_seqs[-1]
            
            #########  update pose ###########
            pose_list_seqs = [None] * len(pose_list)
            for i, (pose, hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                hidden_p, pose_seqs = self.update_block_pose(hidden_p, pose_cost_func_list[i],
                                                             pose, inp_p_list[i], seq_len=self.seq_len)
                hidden_p_list[i] = hidden_p
                if not self.inter_sup:
                    pose_seqs = [pose_seqs[-1]]
                pose_list_seqs[i] = pose_seqs
                
            for pose_list_i in zip(*pose_list_seqs):
                pose_predictions.append([pose.clone() for pose in pose_list_i])

            pose_list = list(zip(*pose_list_seqs))[-1]
            

        if not self.training:
            return inv_depth_predictions[-1], \
                   torch.stack(pose_predictions[-1], dim=1).view(target_image.shape[0], len(ref_imgs), 6) #(b, n, 6)
                
        return inv_depth_predictions, \
               torch.stack([torch.stack(poses_ref, dim=1) for poses_ref in pose_predictions], dim=2) #(b, n, iters, 6)