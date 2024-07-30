"""
Author: Zhiyuan Zhang
Date: July 2024
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import os
import sys
import torch
import numpy as np
import datetime
import logging
import provider
import importlib
import shutil
import argparse
import sklearn.metrics as metrics

from pathlib import Path
from tqdm import tqdm
from data_utils.FG3D_DataLoader import FG3D_DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='risurconv_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--category', default="airplane", type=str, choices=["airplane","car","chair"],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=900, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--process_data', type=bool, default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    parser.add_argument('--pretrain_weight', type=str, default=None)
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



def test(net, testloader):
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    
    with torch.no_grad():
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

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification_FG3D')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.category)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../data/FG3D/'

    train_dataset = FG3D_DataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = FG3D_DataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
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
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model.split('_')[0], str(exp_dir))
    shutil.copy('./train_classification_FG3D.py', str(exp_dir))

    classifier = model.get_model(num_class, 2, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    
    if args.pretrain_weight:
        print("Load initial weight from %s\n" % args.pretrain_weight)
        checkpoint = torch.load(args.pretrain_weight)
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        for name, param in classifier.named_parameters():
            if name in checkpoint['model_state_dict']:
                print(name)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    mat_lr=[]
    mat_test=[]
    mat_cla_test=[]
    mat_train=[]

    log_string('Trainable Parameters: %f' % (count_parameters(classifier)))

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            
            if len(pred.shape) == 3:
                target_2 = target.unsqueeze(-1).repeat(1,pred.shape[1])
                target_2 = target_2.view(-1, 1)[:, 0]
                pred_2 = pred.contiguous().view(-1, num_class)  # N*K, num_class
                loss = criterion(pred_2, target_2.long())
                pred = pred.mean(dim=1) # N, num_class
            else:
                loss = criterion(pred, target.long())

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        if epoch==300:   # Warm up, a popular transformer training scheme
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
        elif epoch==600: # Warm up, a popular transformer training scheme
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate*0.8
        
        
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train loss: %f' % loss)
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])
        mat_lr.append(optimizer.param_groups[0]['lr'])
        mat_train.append(train_instance_acc)
        scheduler1.step()

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            mat_test.append(instance_acc)
            mat_cla_test.append(class_acc)



            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
  

if __name__ == '__main__':
    args = parse_args()
    main(args)