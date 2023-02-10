import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from net_utils import conv
import torch
import torch.nn as nn
from collections import OrderedDict


class FeaturePyramid(nn.Module):
    def __init__(self, hidden_chs=128, out_chs=None,input_images=1):
        super(FeaturePyramid, self).__init__()
        self.out_chs = out_chs
        self.conv1 = conv(3*input_images,   16, kernel_size=3, stride=2)
        self.conv2 = conv(16,  16, kernel_size=3, stride=1)
        self.conv3 = conv(16,  32, kernel_size=3, stride=2)
        self.conv4 = conv(32,  32, kernel_size=3, stride=1)
        self.conv5 = conv(32,  64, kernel_size=3, stride=2)
        self.conv6 = conv(64,  64, kernel_size=3, stride=1)
        self.conv7 = conv(64,  96, kernel_size=3, stride=2)
        self.conv8 = conv(96,  96, kernel_size=3, stride=1)
        self.conv9 = conv(96, 128, kernel_size=3, stride=2)
        self.conv10 = conv(128, 128, kernel_size=3, stride=1)
        self.conv11 = conv(128, 196, kernel_size=3, stride=2)
        self.conv12 = conv(196, 196, kernel_size=3, stride=1)
        if self.out_chs != None:
            self.out = nn.Sequential(OrderedDict(
                [
                    ("conv2-1", nn.Conv2d(hidden_chs, out_chs, kernel_size=3, stride=1, padding=1)),
                    # ("conv2-2", nn.Conv2d(hidden_chs, out_chs, kernel_size=3, stride=2)),
                    ("bn2", nn.BatchNorm2d(out_chs)),
                    ("relu", nn.LeakyReLU(inplace=True))
                ]
            ))

        state_dict = torch.load('weights/kitti_flow.pth')['model_state_dict']
        self.state_dict = {}
        for key, value in state_dict.items():
            if 'fpyramid' in key:
                str = key.split('fpyramid.')
                self.state_dict[str[-1]] = value

        self.state_dict['conv1.0.weight'] = torch.cat([self.state_dict['conv1.0.weight']] * input_images, 1) / input_images


        self.load_state_dict(self.state_dict, strict=False)
        

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.constant_(m.weight.data, 0.0)
                if m.bias is not None:
                    m.bias.data.zero_()
        '''
    def forward(self, img):
        cnv2 = self.conv2(self.conv1(img))
        cnv4 = self.conv4(self.conv3(cnv2))
        cnv6 = self.conv6(self.conv5(cnv4))
        if self.out_chs != None:
            # x = nn.functional.interpolate(cnv6, scale_factor=2, mode='bilinear')
            return self.out(cnv6)
        cnv8 = self.conv8(self.conv7(cnv6))
        cnv10 = self.conv10(self.conv9(cnv8))
        cnv12 = self.conv12(self.conv11(cnv10))
        return cnv2, cnv4, cnv6, cnv8, cnv10, cnv12

