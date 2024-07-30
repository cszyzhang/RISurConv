"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from pointops.functions import pointops

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    dists = torch.cdist(new_xyz, xyz)
    if radius is not None:
        group_idx[dists > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def compute_LRA_one(group_xyz, weighting=False):
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True) # nn lengths
    
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)
    
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    
    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3
    
def compute_LRA(xyz, weighting=False, nsample = 64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    # eigen_values, vec = M.symeig(eigenvectors=True)
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')

    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(
        sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def sample(npoint, xyz, norm=None, sampling='fps'):
    B, N, C = xyz.shape
    xyz = xyz.contiguous()
    if sampling=='fps':
        fps_idx = pointops.furthestsampling(xyz, npoint).long()
        
        new_xyz = index_points(xyz, fps_idx)
        if norm is not None:
            new_norm = index_points(norm, fps_idx)
    elif sampling == 'random':
        shuffle = np.arange(xyz.shape[1])
        np.random.shuffle(shuffle)
        new_xyz = xyz[:, shuffle[:npoint], :]
        if norm is not None:
            new_norm = norm[:, shuffle[:npoint], :]
    else:
        print('Unknown sampling method!')
        exit()
    
    return new_xyz, new_norm

def group_index(nsample, radius, xyz, new_xyz, group='knn'):
    if group == 'knn':
        idx = knn_point(nsample, xyz, new_xyz.contiguous())


    elif group == 'ball':
        idx = pointops.ballquery(radius, nsample, xyz, new_xyz.contiguous())
        idx = idx.long()
    else:
        print('Unknown grouping method!')
        exit()

    return idx

def order_index(xyz, new_xyz, new_norm, idx):
    epsilon=1e-7
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane*new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / (proj_xyz_length)
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero
    

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1,1,1,3)) # corresponds to the largest length
    
    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the vec_ref itself, should be 1.
    dots = sign*dots - (1-sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered

def caculate_distance(xi):
    B,N,num,C=xi.shape
    xi1=torch.unsqueeze(xi,dim=2) #B,N,1,num,C
    xi2=torch.unsqueeze(xi,dim=3) #B,N,num,1,C
    res_xi=xi1-xi2
    e=1e-16
    xi_distance=torch.sqrt(torch.sum(res_xi**2,dim=-1)+e)
    return xi_distance

def calculate_two_surface_feature(x1,x1_norm,x2,x2_norm):
    #feature between surface x1 and surface x2
    epsilon=1e-7
    offest=x1-x2
    surface_offest_length = torch.norm(offest, dim=-1, keepdim=True) # x12 lengths
    offest_x12_unit = offest / (surface_offest_length+epsilon) #d norm as xp
    offest_x12_unit[offest_x12_unit != offest_x12_unit] = 0  # set nan to zero
    

    sin_angle_1 = -(offest_x12_unit * x1_norm).sum(-1, keepdim=True) #cos(pi/2-theta)=sin
    sin_angle_2 =  (offest_x12_unit * x2_norm).sum(-1, keepdim=True) 

    
    return sin_angle_1,sin_angle_2,surface_offest_length

def calculate_unit(new_xi,x1):
    epsilon=1e-7
    offest_xi=new_xi-x1
    surface_offest_length = torch.norm(offest_xi, dim=-1, keepdim=True) # x12 lengths
    offest_xi_unit = offest_xi / (surface_offest_length+epsilon) #d norm as xp
    offest_xi_unit[offest_xi_unit != offest_xi_unit] = 0  # set nan to zero
    
    return offest_xi_unit

def calculate_surface_norm(surface_norm1,surface_norm2):
    norm_x=surface_norm1[:,:,:,1]*surface_norm2[:,:,:,2]-surface_norm1[:,:,:,2]*surface_norm2[:,:,:,1]
    norm_y=surface_norm1[:,:,:,2]*surface_norm2[:,:,:,0]-surface_norm1[:,:,:,0]*surface_norm2[:,:,:,2]
    norm_z=surface_norm1[:,:,:,0]*surface_norm2[:,:,:,1]-surface_norm1[:,:,:,1]*surface_norm2[:,:,:,0]
    norm=torch.cat([norm_x.unsqueeze(-1),norm_y.unsqueeze(-1),norm_z.unsqueeze(-1)],dim=-1)
    return norm

def calculate_new_surface_feature(new_xi,new_xi_norm,x1,x1_norm,x2,x2_norm,x3,x3_norm):
    #feature between surface x1 and surface x2

    pxi_u=calculate_unit(new_xi,x1)
    px2_u=calculate_unit(x2,x1)
    x2xi_u=calculate_unit(new_xi,x2)
    px3_u=calculate_unit(x3,x1)

    surface_norm1=calculate_surface_norm(pxi_u,px2_u)
    surface_norm2=calculate_surface_norm(px3_u,px2_u)

    sin_angle_1_1 = (pxi_u * px2_u).sum(-1, keepdim=True) #cos(pi/2-theta)=sin
    sin_angle_1_2 = (pxi_u * x2xi_u).sum(-1, keepdim=True) 

    sin_angle_3=(surface_norm1 * surface_norm2).sum(-1, keepdim=True) 

    sin_angle_2_1=( x2xi_u * new_xi_norm).sum(-1, keepdim=True)
    sin_angle_2_2=( px2_u * new_xi_norm).sum(-1, keepdim=True)
 
    new_feature = torch.cat([
            sin_angle_1_1,
            sin_angle_1_2,
            sin_angle_2_1,       
            sin_angle_2_2,
            sin_angle_3
            ], dim=-1)

    return new_feature

def RISP_features(xyz, norm, new_xyz, new_norm, idx, group_all=False):
    B, N, C = new_xyz.shape
    num_neighbor = idx.shape[-1]
    dots_sorted, idx_ordered = order_index(xyz, new_xyz, new_norm.unsqueeze(-1), idx)
    
    grouped_center = index_points(xyz, idx_ordered)  # [B, npoint, nsample, C]
    xi_norm=index_points(norm, idx_ordered)             # xi norm
    if not group_all:
        xi = grouped_center - new_xyz.view(B, N, 1, C)  # xi
    else:
        xi = grouped_center                             # xi
    
    p_point = torch.zeros_like(xi)                                # p
    p_norm=(new_norm.unsqueeze(-2)).repeat([1,1,num_neighbor,1])  # p norm

    num_shifts = 1
    if N>=1024:
        num_shifts = 2
    
    x3=torch.roll(xi,shifts=num_shifts,dims=2)              # xi-1
    x3_norm=torch.roll(xi_norm,shifts=num_shifts,dims=2)    # xi-1 norm

    sin_angle_1_0,sin_angle_2_0, length_0=calculate_two_surface_feature(p_point,p_norm,xi,xi_norm)  # alpha1 beta 1
    sin_angle_1_1,sin_angle_2_1, length_1=calculate_two_surface_feature(p_point,p_norm,x3,x3_norm)  # alpha2 theta 1
    sin_angle_1_2,sin_angle_2_2, length_2=calculate_two_surface_feature(xi,xi_norm,x3,x3_norm)      # beta 2 theta 2

    #############
    angle_0 = (calculate_unit(p_point,xi) * calculate_unit(p_point,x3)).sum(-1, keepdim=True)
    angle_1 = (calculate_unit(x3,p_point) * calculate_unit(x3,xi)).sum(-1, keepdim=True)

    ri_feat = torch.cat([
            length_0,
            sin_angle_1_0,
            sin_angle_2_0,
            angle_0, 
            sin_angle_1_1,
            sin_angle_2_1,
            angle_1,
            sin_angle_1_2,
            sin_angle_2_2,
            ], dim=-1)
        
    
    x4 = torch.roll(xi,shifts=-num_shifts,dims=2)  #xi+1
    x4_norm = torch.roll(xi_norm,shifts=-num_shifts,dims=2)#xi+1 norm
    new_feature=calculate_new_surface_feature(x4,x4_norm,p_point,p_norm,xi,xi_norm,x3,x3_norm)
    ri_feat = torch.cat([ri_feat, new_feature], dim=-1)

    return ri_feat, idx_ordered

def sample_and_group(npoint, radius, nsample, xyz, norm):
    """
    Input:
        npoint: number of new points
        radius: radius for each new points
        nsample: number of samples for each new point
        xyz: input points position data, [B, N, 3]
        norm: input points normal data, [B, N, 3]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        ri_feat: sampled ri attributes, [B, npoint, nsample, 8]
        new_norm: sampled norm data, [B, npoint, 3]
        idx_ordered: ordered index of the sample position data, [B, npoint, nsample]
    """
    xyz = xyz.contiguous()
    norm = norm.contiguous()
 
    new_xyz, new_norm = sample(npoint, xyz, norm=norm, sampling='fps')
    idx = group_index(nsample, radius, xyz, new_xyz, group='knn')
    
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)

    
    return new_xyz, ri_feat, new_norm, idx_ordered
    
def sample_and_group_all(xyz, norm):

    device = xyz.device
    B, N, C = xyz.shape
    S=1
    new_xyz = torch.mean(xyz, dim=1, keepdim=True) # centroid
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered
    new_norm = compute_LRA_one(grouped_xyz_local, weighting=True)
    
    
    device = xyz.device
    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # create ri features
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx, group_all=True)

    return None, ri_feat, new_norm, idx_ordered

def sample_and_group_deconv(nsample, xyz, norm, new_xyz, new_norm):
    idx = group_index(nsample, 0.0, xyz, new_xyz, group='knn')
    ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)

    return ri_feat, idx_ordered

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        #self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class SA_Layer_2d(nn.Module):
    def __init__(self, channels):
        super(SA_Layer_2d, self).__init__()
        self.q_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.trans_conv = nn.Conv2d(channels, channels, 1)
        self.after_norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 3, 2, 1)  # b, points, n, c 
        x_k = self.k_conv(x).permute(0, 3, 1, 2)  # b, points, c, n        
        x_v = self.v_conv(x).permute(0, 3, 1, 2)
        energy = x_q @ x_k    # b,points, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b,points, c, n 
        x_r=x_r.permute(0,2,3,1)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class RISurConvSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, out_channel, group_all):
        super(RISurConvSetAbstraction, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        raw_in_channel= [14, 32]
        raw_out_channel=[32, 64]
        
        self.embedding=nn.Sequential(
            nn.Conv2d(raw_in_channel[0], raw_out_channel[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[0]),
            nn.Conv2d(raw_in_channel[1], raw_out_channel[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[1])
        )
        self.self_attention_0 = SA_Layer_2d(raw_out_channel[1])  # 1st SA layer

        self.risurconv=nn.Sequential(
            nn.Conv2d(raw_out_channel[1] + in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.self_attention_1 = SA_Layer(out_channel) # 2nd SA layer

    def forward(self, xyz, norm, points):
        """
        Input:
            surface_data: input surface center point position data, [B, N, 6]
            feature: input old feature data, [B, C, N]
        Return:
            surface data : sampled new surface data, [B, new_N, 6]
            feature: output new feature data, [B, C, new_N]
        """

        if points is not None:  # transform from [B, C, N] to [B, N, C]
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
  
        if self.group_all:
            new_xyz, ri_feat, new_norm, idx = sample_and_group_all(xyz, norm)
        else:
            new_xyz, ri_feat, new_norm, idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz, norm)

        # embed
        ri_feat=F.relu(self.embedding(ri_feat.permute(0, 3, 2, 1)))  

        ri_feat = self.self_attention_0(ri_feat)

        # concat previous layer features
        if points is not None:
            if idx is not None:
                grouped_points = index_points(points, idx)
            else:
                grouped_points = points.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        new_points = F.relu(self.risurconv(new_points))

        risur_feat = torch.max(new_points, 2)[0]  # maxpooling
        risur_feat = self.self_attention_1(risur_feat)
        
        return new_xyz, new_norm, risur_feat



class RIConv2FeaturePropagation(nn.Module):
    def __init__(self, radius, nsample, in_channel, in_channel_2, out_channel, mlp):
        super(RIConv2FeaturePropagation, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # lift to 64
        raw_in_channel= [14, 32]
        raw_out_channel=[32, 64]
        
        self.embedding=nn.Sequential(
            nn.Conv2d(raw_in_channel[0], raw_out_channel[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[0]),
            nn.Conv2d(raw_in_channel[1], raw_out_channel[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[1])
        )

        self.self_attention_0 = SA_Layer_2d(raw_out_channel[1])  # 1st SA layer

        #concat previous layer features
        self.risurconv=nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.self_attention_1 = SA_Layer(out_channel) # 2nd SA layer

        # mlp block
        last_channel = in_channel_2
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, norm1, norm2, points1, points2):

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

                   
        ri_feat, idx_ordered = sample_and_group_deconv(self.nsample, xyz2, norm2, xyz1, norm1)
        # embeding
        ri_feat=F.relu(self.embedding(ri_feat.permute(0, 3, 2, 1)))  
        ri_feat = self.self_attention_0(ri_feat)  # 1st SA layer

        # concat previous layer features
        if points2 is not None:
            if idx_ordered is not None:
                grouped_points = index_points(points2, idx_ordered)
            else:
                grouped_points = points2.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1) # [B, npoint, nsample, C+D]
        else:
            new_points = ri_feat

        # risurconv
        new_points = F.relu(self.risurconv(new_points))
        new_points = torch.max(new_points, 2)[0]  # maxpooling
        new_points = self.self_attention_1(new_points)

        # mlp block
        if points1 is not None:
            new_points = torch.cat([new_points, points1], dim=1)
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points = F.relu(bn(conv(new_points)))

        return new_points
    
if __name__ == '__main__':
    nsample=64
    ref=torch.rand(16,100,3).cuda()
    query=torch.rand(16,20,3).cuda()

    start=time()
    for i in range(50):
        idx = knn_point(nsample, ref, query.contiguous())
    print(time()-start)

    start=time()
    for i in range(50):
        idx = pointops.knnquery(nsample, ref, query.contiguous())
    print(time()-start)