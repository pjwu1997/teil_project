from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from NetworkInNetwork import Regressor
from NonLinearClassifier import Classifier
import numpy as np
from torch import Tensor

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--dataroot', default='./cifar', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--net', default='/home/zeus/pj/AET/cifar/projective/output/net_epoch_1499.pth', type=str, metavar='PATH',
                    help='path to the trained unsupervised model (default: none)')                    
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    
best_acc = 0  # best test accuracy

def main():
    global best_acc
    global device
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset cifar10')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataloader = datasets.CIFAR10
    num_classes = 10

    trainset = dataloader(root=args.dataroot, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, num_workers=args.workers)

    # testset = dataloader(root=args.dataroot, train=False, download=False, transform=transform_test)
    # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    net = Regressor(_num_stages=4, _use_avg_on_conv3=False).to(device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    if args.net != '':
        print('Yes')
        net.load_state_dict(torch.load(args.net))

    cudnn.benchmark = True

    latent_lst = get_latent_vector(trainloader, net, use_cuda, device)

    # with open('data.json', 'w') as f:
    #     json.dump(latent_lst, f)


def get_latent_vector(trainloader, net, use_cuda, device):
    # switch to train mode
    net.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    return_lst = []
    with open('data.json', 'w+') as f:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # measure data loading time
            # print('yoyo')
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)

            # compute output
            f1, f2 = net(inputs, inputs, out_feat_keys=['conv2'])
            tmp = f1.cpu().detach().numpy().flatten()
            np.savetxt(f, tmp, newline=" ")
            print("\n", file=f)

def concate():
    
    with open('../../../data.txt', 'r') as f:
    #with open('777.txt', 'r') as f:
        lines = f.readlines()
        print(lines)
        
        data1 = [[line] for line in lines]
        # print(data1[0])
        data1 = np.array(data1)
        print(data1.shape())

if __name__ == '__main__':
    main()
