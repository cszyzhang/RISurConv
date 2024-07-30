"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""
import os
import torch
from data_utils.FG3D_DataLoader import FG3D_DataLoader
import argparse
import numpy as np
import logging
from tqdm import tqdm
import sys
import importlib
import sklearn.metrics as metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--category', default="car", type=str, help='training on FG3D [airplane, chair, car]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pretrained', help='log root')
    parser.add_argument('--process_data', type=bool, default=True, help='save data offline')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    return parser.parse_args()


def test(net, testloader):
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    
    
    for batch_idx, (data, label) in tqdm(enumerate(testloader), total=len(testloader)):
        logits, feat = net(data.cuda())
        if len(logits.shape) == 3:
            logits = logits.mean(dim=1)

        preds = logits.data.max(1)[1].cpu()
            
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    instance_acc = metrics.accuracy_score(test_true, test_pred)
    class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    experiment_dir = 'log/classification_FG3D/' + args.category + '/' + args.log_dir
    print(experiment_dir)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../../data/FG3D/' # original data
    if args.process_data:
        data_path = '../../data/FG3D/' # preprocessed data
    
    
    test_dataset = FG3D_DataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    '''MODEL LOADING'''
    if args.category=="airplane":
        num_class = 13
    elif args.category=="car":
        num_class = 20
    elif args.category=="chair":
        num_class = 33
    else:
        print("wrong num_class input")
    model = importlib.import_module('risurconv_cls')

    classifier = model.get_model(num_class, 2)
    classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

if __name__ == '__main__':
    args = parse_args()
    main(args)