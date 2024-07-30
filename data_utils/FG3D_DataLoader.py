import os
import numpy as np
import warnings
import pickle
import open3d as o3d
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class FG3D_DataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=True):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.use_normals = args.use_normals
        self.category = args.category
        
        if self.category == "airplane":
            self.catfile = os.path.join(self.root, 'Airplane_subcategories.txt')
        elif self.category == "car" :
            self.catfile = os.path.join(self.root, 'Car_subcategories.txt')
        elif self.category == "chair":
            self.catfile = os.path.join(self.root, 'Chair_subcategories.txt')      
        else :
            print("wrong category")
        print("using area: "+args.category )

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.category == "airplane":
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'Airplane_off_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'Airplane_off_test.txt'))]
        elif self.category == "car":
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'Car_off_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'Car_off_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'Chair_off_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'Chair_off_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, self.category+"_off", shape_ids[split][i]) + '.off') for i
                        in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.save_path = os.path.join(root, 'FG3D_%s_%s_%dpts_fps_processed.dat' % (self.category, split, self.npoints))
        if self.process_data and os.path.exists(self.save_path):
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)          
        else:
            if self.process_data and not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
            elif not self.process_data:
                print('Reprocessing data %s ...' % self.save_path)
                print('Make sure that there is no files in this path(%s)' % self.save_path)
                if os.path.exists(self.save_path):
                    raise ValueError("There are files in the offline save path")
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)
            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)              
                mesh = o3d.io.read_triangle_mesh(fn[1])
                pcd = mesh.sample_points_poisson_disk(number_of_points=self.npoints, init_factor=5, use_triangle_normal=True)
                points = np.asarray(pcd.points) # shape = (n, 3)
                if self.use_normals:
                    normals = np.asarray(pcd.normals)
                    point_set = np.concatenate((points, normals), axis=-1) # shape = (n, 6)
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)   

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set=np.array(point_set)   
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set.astype(np.float32), label[0]

    def __getitem__(self, index):
        return self._get_item(index)
