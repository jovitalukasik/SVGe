import argparse
import glob
import json
import logging
import os
import sys
import time
import random 
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.autograd import Variable
from DownsampledImageNet import ImageNet16


import torchvision
import torchvision.transforms as transforms

from model import Network
from model_spec import ModelSpec
import utils 
from shutil import copy


parser = argparse.ArgumentParser("Train NASBench-101 cell like on Cifar10 or ImageNet16-120")
parser.add_argument('--name',                   type=str ,default='debugging')
parser.add_argument('--data',                   type=str, default='data', help='location of the cifar10 data')
parser.add_argument('--data_set',               default='cifar10', help= 'which dataset cifar10 or Imagenet16')
parser.add_argument('--batch_size',             type=int, default=128, help='batch size, 256 for Imagenet16')
parser.add_argument('--learning_rate',          type=float, default=0.2, help='init learning rate')  # Increased learning rate
parser.add_argument('--lr_decay_method',        default='COSINE_BY_STEP', type=str, help='learning decay method')
parser.add_argument('--learning_rate_min',      type=float, default=0, help='min learning rate')
parser.add_argument('--momentum',               type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay',           type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq',            type=float, default=50, help='report frequency')
parser.add_argument('--device',                 type=str, default='cuda:1')
parser.add_argument('--epochs',                 type=int, default=108, help='num of training epochs, 200 for Imagenet')
parser.add_argument('--cutout',                 action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length',          type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob',            type=float, default=1.0, help='cutout probability')
parser.add_argument('--grad_clip',              type=float, default=5, help='gradient clipping')
parser.add_argument('--val_portion',            type=float, default=0.8, help='portion of valdiation set, 0.5 for ImageNet')
parser.add_argument('--seed',                   type=int, default=0, help='Set random seed')

args = parser.parse_args()

seed = args.seed
print(f"Random Seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = args.device

##############################################################################
#
#                              Runfolder
#
##############################################################################
#Create Log Directory
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"{args.data_set}/{runfolder}_{args.name}"

FOLDER_EXPERIMENTS = os.path.join(os.getcwd(), 'Experiments/Train_Cell/')
log_dir = os.path.join(FOLDER_EXPERIMENTS, runfolder)
os.makedirs(log_dir)

# Dump the config of the run
with open(os.path.join(log_dir, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

CIFAR_CLASSES = 10
IMAGENET_CLASSES= 120

matrix=[[0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0]]  
operations=['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output']

def main():
    
    logging.info("Cell to be trained= {}{}".format(np.array(matrix),operations))
    with open(os.path.join(log_dir, 'cell.txt'), 'a') as file:
        json.dump(matrix, file)
        file.write('\n')
        json.dump(operations, file)
        file.write('\n')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    
    if args.data_set == 'cifar10': 
        num_classes=CIFAR_CLASSES
    elif args.data_set == 'ImageNet16':
        num_classes=IMAGENET_CLASSES
    else:
        raise TypeError("Unknow dataset : {:}".format(args.data_set))



    ##############################################################################
    #
    #                                 Model 
    #
    ##############################################################################

    spec = _ToModelSpec(matrix, operations)
    model = Network(spec, num_classes)

    model = model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        eps=1.0,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    ##############################################################################
    #
    #                              Dataset
    #
    ##############################################################################
    print("Creating Dataset.")

    # cifar10 dataset
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.data_set == 'cifar10': 
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        test_data=dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.val_portion * num_train))
        
        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True)
        valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), 
                                                pin_memory=True)
        test_queue=torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        logging.info('Training set size: {}, Validation set size: {}, Test set size: {}'.format(split, num_train-split, len(test_data)))

    elif args.data_set == 'ImageNet16':
        train_transform, valid_transform = utils._data_transforms_ImageNet16(args)
        train_data = ImageNet16(root=os.path.join(dir_path, args.data), train=True ,transform= train_transform, use_num_of_class_only=120)
        test_data  = ImageNet16(root=os.path.join(dir_path, args.data), train=False, transform=valid_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700 and len(test_data) == 6000
        num_test = len(test_data)
        indices = list(range(num_test))
        split = int(np.floor(args.val_portion * num_test))
    
        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        imagenet16_splits = utils.load_config(os.path.join(dir_path, 'configs/imagenet-16-120-test-split.txt'), None, None)
        valid_queue=torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet16_splits.xvalid), num_workers=2, pin_memory=True)
        test_queue=torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet16_splits.xtest ), num_workers=2, pin_memory=True)

        logging.info('Training set size: {}, Validation set size: {}, Test set size: {}'.format(len(train_data), num_test-split, split))


    else:
         raise TypeError("Unknow dataset : {:}".format(args.data_name))




    ##############################################################################
    #
    #                              Training
    #
    ##############################################################################

    budget = args.epochs
    for epoch in range(budget):
        lr = scheduler.get_lr()[0]

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, lr, epoch, device)
        logging.info('epoch %03d train_acc %f', epoch, train_acc)
        

         # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, device)
        logging.info('epoch %03d valid_acc %f', epoch, valid_acc)
        
        # test
        test_acc, test_obj = test(test_queue, model, criterion, device)
        logging.info('epoch %03d test_acc %f', epoch, test_acc)
        
        scheduler.step()

        # Save the entire model
        if epoch % 50 == 0:
            filepath = os.path.join(log_dir, 'model_{}.obj'.format(epoch))
            torch.save(model.state_dict(), filepath)

        config_dict = {
            'epochs': epoch,
            'train_acc': train_acc,
            'valid_acc':valid_acc,
            'test_acc': test_acc,
            }  

        with open(os.path.join(log_dir, 'results.txt'), 'w') as file:
                json.dump(str(config_dict), file)
                file.write('\n')

        

            

def train(train_queue, valid_queue, model, criterion, optimizer, lr, epoch, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    
    
    correct = 0
    total = 0
    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
        
        n = inputs.size(0)


        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        ##from nasbench pytorch
        total += n
        _, predict = torch.max(logits.data, 1)
        correct += predict.eq(targets.data).cpu().sum().item()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss: %.3f top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)
    logging.info('Epoch=%d  | Acc=%.3f(%d/%d)' %
                  (epoch, correct/total, correct, total))


    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    correct = 0
    total=0
    for step, (inputs, targets) in enumerate(valid_queue):
        with torch.no_grad():
            inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))

            logits = model(inputs)
            loss = criterion(logits, targets)
        
        total +=  inputs.size(0)
        _, predict = torch.max(logits.data, 1)
        correct += predict.eq(targets.data).cpu().sum().item()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d loss: %.3f top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)
    logging.info('validation:  Acc=%.3f(%d/%d)' %
              (correct/total, correct, total))



    return top1.avg, objs.avg


def test(test_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    correct = 0
    total=0
    for step, (inputs, targets) in enumerate(test_queue):
        with torch.no_grad():
            inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
            logits = model(inputs)
            loss = criterion(logits, targets)
        
        total +=  inputs.size(0)
        _, predict = torch.max(logits.data, 1)
        correct += predict.eq(targets.data).cpu().sum().item()

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d loss: %.3f top1: %f top5: %f', step, objs.avg, top1.avg, top5.avg)
    logging.info('Testing:  Acc=%.3f(%d/%d)' %
              (correct/total, correct, total))



    return top1.avg, objs.avg



def _ToModelSpec(mat, ops):
    return ModelSpec(mat, ops)

if __name__ == '__main__':
    main()
