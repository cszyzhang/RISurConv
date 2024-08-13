# RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation

European Conference on Computer Vision (**ECCV 2024**) **Oral** |  [PDF](https://arxiv.org/pdf/2408.06110)

[Zhiyuan Zhang](https://zhiyuanzhang.net/), Licheng Yang, Zhiyu Xiang.

If you found this paper useful in your research, please cite:
```
@inproceedings{zhang2024risurconv,
  title={RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation},
  author={Zhang, Zhiyuan and Yang, Licheng and Xiang, Zhiyu},
  booktitle={2024 European Conference on Computer Vision (ECCV)},
  pages={1--14},
  year={2024}
}
```

## Installation
This repo provides the RISurConv source codes, which had been tested with Python 3.10.14, PyTorch 1.9.0, CUDA 12.1 on Ubuntu 20.04.  

Install the pointnet++ cuda operation library by running the following command:
```
cd pointops
python3 setup.py install
cd ..
```

## Usage
### Pretrained Models
We provide pretrained models for all the experiments in this work. Please download the zip file [**HERE**](https://1drv.ms/u/c/1cd2dc535b9bd761/EexVLK4B1hNHs-7VLxIPNzUBAwJuvvnV5esXl7iCRuhbNQ?e=1poaOn). The zip file contains the pretrained models as well as the training log files. Unzip the log fold under the project folder. Make sure that folder architectures are as follows:
```
│data/
├── FG3D
├── modelnet40_preprocessed/
├── scanobjectnn/
├── shapenetcore_partanno_segmentation_benchmark_v0_normal/
│RISurConv/
├── data_utils/
├── log/
│   ├── classification_FG3D/
│   ├── classification_modelnet40/
│   ├── classification_scanobj/
│   ├── classification_partseg/
├── models/
├── pointops/
├── test_classification_FG3D.py
├── test_classification_modelnet40.py
├── test_classification_scanobj.py
├── test_partseg.py
├── train_classification_FG3D.py
├── train_classification_modelnet40.py
├── train_classification_scanobj.py
├── train_partseg.py
```

### Classification
We perform classification on FG3D, ModelNet40 and ScanObjectNN respectively.
#### FG3D

Download the **FG3D dataset** [here](https://github.com/liuxinhai/FG3D-Net) and save the file into `../data/FG3D/`. The origianl data format is mesh, please use `--process_data` to preprocess the data to extract the point cloud and the corresponding normal vectors, and put the processed data to `../data/FG3D/`. Alternatively, you can also download the pre-processd data [here](https://1drv.ms/u/c/1cd2dc535b9bd761/AUaS8N7HHHFPrQ2zKXdhXZY?e=f0qFxb) and unzip it in `../data/FG3D/`. (**Note**: the `data/` folder is outside the project folder)

There are 3 categories in FG3D dataset: Airplane, Chair, Car. To train a RISurConv model to classify object in the **airplane** category:
```
python3 train_classification_FG3D.py --category 'airplane' --epoch 300 --decay_rate 1e-2
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/classification_FG3D/airplane/pretrained/checkpoints/`):
```
python3 test_classification_FG3D.py --category 'airplane' --log_dir 'pretrained'
```

To train a RISurConv model to classify object in the **chair** category:
```
python3 train_classification_FG3D.py --category 'chair'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/classification_FG3D/chair/pretrained/checkpoints/`):
```
python3 test_classification_FG3D.py --category 'chair' --log_dir 'pretrained'
```

To train a RISurConv model to classify object in the **car** category:
```
python3 train_classification_FG3D.py --category 'car'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/classification_FG3D/car/pretrained/checkpoints/`):
```
python3 test_classification_FG3D.py --category 'car' --log_dir 'pretrained'
```

#### ModelNet40

Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `../data/modelnet40_normal_resampled/`. Follow the instructions of [PointNet++(Pytorch)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) to prepare the data. Specifically, please use `--process_data` to preprocess the data, and move the processed data to `../data/modelnet40_preprocessed/`. Alternatively, you can also download the pre-processd data [here](https://1drv.ms/u/s!AmHXm1tT3NIcnnBiRlVxATXtOhe9?e=oynmh2) and save it in `../data/modelnet40_preprocessed/`. (**Note**: the `data/` folder is outside the project folder)

To train a RISurConv model to classify shapes in the ModelNet40 dataset:
```
python3 train_classification_modelnet40.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/classification_modelnet40/pretrained/checkpoints/`):
```
python3 test_classification_modelnet40.py --log_dir 'pretrained'
```
#### ScanObjectNN
Download the **ScanObjectNN** [here](https://hkust-vgd.github.io/scanobjectnn/) and save the `main_split` and `main_split_nobg` subfolders that inlcude the h5 files into the `../data/scanobjectnn/` (**Note**: the `data/` folder is outside the project folder)

Training on the hardest variant **PB_T50_RS**:
```
python3 train_classification_scanobj.py --data_type 'hardest'
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/classification_scanobj/pretrained/checkpoints/`):
```
python3 test_classification_scanobj.py --data_type 'hardest' --log_dir 'pretrained'
```

### Segmentation
We perform part segmentation and semantic segmentation on ShapeNet and S3DIS respectively.

#### ShapeNet
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. (**Note**: the `data/` folder is outside the project folder)

Training:
```
python3 train_partseg.py
```
For testing, you can use your trained model by specifying `--log_dir` or use our **pretrained model** directly (make sure the pretrained best_model.pth is in `log/partseg/pretrained/checkpoints/`):
```
python3 test_partseg.py --log_dir 'pretrained'
```

## License
This repository is released under MIT License (see LICENSE file for details).
