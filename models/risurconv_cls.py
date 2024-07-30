"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch.nn as nn
import torch.nn.functional as F
from models.risurconv_utils import RISurConvSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_class, n, normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel
        self.sc0 = RISurConvSetAbstraction(npoint=512*n, radius=0.12, nsample=8, in_channel= 0, out_channel=32, group_all=False)
        self.sc1 = RISurConvSetAbstraction(npoint=256*n, radius=0.16, nsample=16, in_channel=32, out_channel=64,  group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=128*n, radius=0.24, nsample=32, in_channel=64, out_channel=128,  group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=64*n, radius=0.48, nsample=64, in_channel=128, out_channel=256,  group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, out_channel=512,  group_all=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8,dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            # compute the LRA and use as normal
            norm = None

        l0_xyz, l0_norm, l0_points = self.sc0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        x=l4_points.permute(0, 2, 1)
        x=self.transformer_encoder(x)
        globle_x = x.view(B, 512)
        # globle_x = torch.max(l4_feature, 2)[0]
        
        x = self.drop1(F.relu(self.bn1(self.fc1(globle_x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        
        return x, l4_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
