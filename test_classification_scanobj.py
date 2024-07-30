"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch
from data_utils.ScanObjectNNLoader import ScanObjectNN
import argparse
import numpy as np
import os
import torch
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
    parser.add_argument('--num_category', default=15, type=int, help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pretrained', help='log root')
    parser.add_argument('--data_type', type=str, default='hardest', help='data type')
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


    '''CREATE DIR'''
    experiment_dir = 'log/classification_scanobj/' + args.log_dir

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
    if args.data_type == 'OBJ_NOBG':
        data_path = '../../data/scanobjectnn/main_split_nobg/'
    elif args.data_type == 'hardest' or 'OBJ_BG': 
        data_path = '../../data/scanobjectnn/main_split/'
    else:
        raise NotImplementedError()
    test_dataset = ScanObjectNN(root=data_path, args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module('risurconv_cls')

    classifier = model.get_model(num_class, 1)
    classifier = classifier.cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

if __name__ == '__main__':
    args = parse_args()
    main(args)
