import torch
import torch.utils.data as data
import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointops.functions import pointops
from models.risurconv_utils import compute_LRA
from models.risurconv_utils import index_points

class ScanObjectNN(Dataset):
    def __init__(self, root, args, split='train'):
        super().__init__()
        assert (split == 'train' or split == 'test')
        self.root = root
        self.num_points = args.num_point
        self.uniform = args.use_uniform_sample
        self.data_type = args.data_type
        
        if split == 'train':
            self.train = True
            if self.data_type == 'hardest':
                h5 = h5py.File(self.root + 'training_objectdataset_augmentedrot_scale75.h5', 'r')
            else:
                h5 = h5py.File(self.root + 'training_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif split == 'test':
            self.train = False
            if self.data_type == 'hardest':
                h5 = h5py.File(self.root + 'test_objectdataset_augmentedrot_scale75.h5', 'r')
            else:
                h5 = h5py.File(self.root + 'test_objectdataset.h5', 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        if self.uniform:
            self.points_ = torch.tensor(self.points).cuda()
            # fps_idx = pointnet2_utils.furthest_point_sample(self.points_, self.num_points)
            fps_idx = pointops.furthestsampling(self.points_, self.num_points)
            self.points_ = index_points(self.points_, fps_idx.long())
            self.points = self.points_.cpu()
            del self.points_
            del fps_idx
            torch.cuda.empty_cache()
        else:
            self.points = self.points[self.num_points, :]
        

        # compute the normal vector based on LRA
        norm = torch.zeros_like(self.points)
        for i in range(self.points.shape[0]):
            norm_i = compute_LRA(self.points[i,:,:].unsqueeze(0), True, nsample = 32)
            norm[i,:,:] = norm_i

        self.points = torch.cat([self.points, norm], dim=-1)
        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.train:
            np.random.shuffle(pt_idxs)     
        current_points = self.points[idx, pt_idxs].clone()
        current_points = current_points.numpy()
        label = self.labels[idx]
        return current_points, label

    def __len__(self):
        return self.points.shape[0]