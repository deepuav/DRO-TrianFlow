'''
Author: Wangpeng
Date: 2022-11-29 17:55:14
LastEditors: Wangpeng-512 1119785034@qq.com
LastEditTime: 2022-12-05 16:22:52
FilePath: \dro-sfm\dro_sfm\trainers\DepthPose.py
Description: 
'''
import pytorch_lightning as pl

class DepthPose(pl.LightningModule): 
    def __init__(self, model, version=None, min_depth=0.1, max_depth=100, **kwargs):
        super(DepthPose, self).__init__()
        self.model = model

    def configure_optimizers(self):
        optimer, lr = self.model.configure_optimizers()
        return [optimer],[lr]

    def forward(self, target_image, ref_imgs, intrinsics):
        batch = {'rgb':target_image, 'rgb_context':ref_imgs, 'intrinsics':intrinsics}
        return self.model.evaluate_depth(batch)

    def training_step(self,batch, batch_idx):
        '''
        batch{
            'idx':idx,
            'filename':image_name,
            'rgb':image,
            intrinsics':intr,
            'pose_context':rel_poses,
            'depth':resized_depth,
            'rgb_context':context_image
        }
        '''
        out = self.model.training_step(batch)
        return out['loss']
    def training_epoch_end(self, out_batch) -> None:
        self.model.training_epoch_end(out_batch)

    def validation_step(self, batch, *args) :
        self.model.validation_step(batch, args)
