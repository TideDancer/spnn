from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
from copy import deepcopy
import Model
from Util import *
import argparse

parser = argparse.ArgumentParser(description='SPNN')
parser.add_argument('arch', metavar='ARCH',
                    help='model architecture: lenet5,vgg16,resnet20')
parser.add_argument('dataset', metavar='DATASET',
                    help='dataset name: MNIST,CIFAR10')
parser.add_argument('path_data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('path_model', metavar='PRETRAINED',
                    help='pretrained model file')
parser.add_argument('-p', '--path_check', default='checkpoint/z.pk', metavar='CHECK',
                    help='path to checkpoint')
parser.add_argument('--save_format', default='net', metavar='FORMAT',
                    help='save format, net,state_dict')
parser.add_argument('target_acc', metavar='TARGETACC', type=float,
                    help='target acc: for comparison, suggest: vgg16(0.911), lenet5(0.9911), resnet(0.909)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-l', '--lr_forward', default=1e-3, type=float,
                    metavar='LRF', help='forward learning rate, default=1e-3 (suggest: resnet 1e-4')
parser.add_argument('--lr_adv', default=1e-2, type=float,
                    metavar='LRA', help='adv learning rate, default=1e-2')
parser.add_argument('-e', '--adv_eps', default=0.1, type=float,
                    metavar='EPS', help='adv bound, default=0.1 (suggest: vgg 5e-4)')
parser.add_argument('-s', '--stop_diff', default=1, type=float,
                    metavar='SD', help='adv training stopping if diff < SD, default=1')
parser.add_argument('--thres', default=0.9, type=float,
                    metavar='TH', help='clip elements < adv_eps*thres, default: 0.9')
parser.add_argument('-T', '--epoch', default=10, type=int,
                    metavar='T', help='forward training epoch, default=10')
parser.add_argument('-I', '--interval', default=5, type=int,
                    metavar='IT', help='learning rate decay (0.5) interval, default=5')
parser.add_argument('-r', '--path_res', default='', metavar='RESUME', 
                    help='resume from checkpoint file, if set will resume training')
parser.add_argument('--res_layer', default=0, type=int, metavar='RESUME', 
                    help='resume from which layer, only useful when resuming mode')

args = parser.parse_args()

# ------------ dataset -----------
if args.dataset == 'CIFAR10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_size = 32
    n_channel = 3
    
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
    
    trainset = torchvision.datasets.CIFAR10(root=args.path_data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.path_data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    advloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
elif args.dataset == 'MNIST':
    img_size = 28
    n_channel = 1
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=False, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    advloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)

elif args.dataset == 'ImageNet':
    img_size = 224
    n_channel = 3
    traindir = args.path_data + 'train'
    valdir = args.path_data + 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
            )
    testloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
            )
    advloader = trainloader

net = torch.load(args.path_res)
net = net.cpu()
print(net)
if args.save_format == 'net': torch.save(net, args.path_check)
else: torch.save(net.state_dict(), args.path_check)


# ------------ main ----------
loss_ce = nn.CrossEntropyLoss()
alpha = 1e-2
ITER_NUM = 10000
LARGE_NUM = 10000
SMALL_NUM = 1e-3


max_val = 2.7537
min_val = -2.4291
val_range = max_val - min_val
R = 0.01
Nb = 10
Ns = 10
max_allowed = val_range * R

def MLE_reverse_Weibull(S):
    return max(S)
      
import numpy as np
from torch.autograd import Variable
def compute_clever(epoch):
    net.eval()
    u_sum = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        #inputs, targets = inputs.cuda(), targets.cuda()
        # optimizer_adv.zero_grad()
        print(batch_idx)
        u = LARGE_NUM
        for j in range(10):
            if j == targets.numpy()[0]: continue
            S = []
            for i in range(Nb):
                b_ik = -LARGE_NUM
                for k in range(Ns):
                    samples = np.random.uniform(low=-max_allowed, high=max_allowed, size=(1, 3, 32, 32))
                    samples = torch.tensor(samples, dtype=torch.float)
                    adv_inputs = inputs + samples
                    adv_inputs = Variable(adv_inputs, requires_grad=True)
                    outputs = F.softmax(net(adv_inputs))
                    loss = outputs[0][targets.numpy()[0]] - outputs[0][j]
                    loss.backward(retain_graph=True)
                    b_ik = max(b_ik, torch.sum(torch.abs(adv_inputs.grad)).numpy()) # 1 norm of vector
                    print(b_ik)
                S.append(b_ik)
            a = MLE_reverse_Weibull(S)  # MLE of a in reverse Weibull
            outputs = net(inputs)
            g_x0 = outputs[0][targets] - outputs[0][j]
            g_x0 = g_x0.detach().numpy()[0]
            u = min(u, min(g_x0/a, R))
        u_sum += u
        print(u)
    return u_sum/batch_idx
 
score = compute_clever(0)
print(score)


