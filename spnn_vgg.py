'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
import argparse
import Model
# from utils import progress_bar
import numpy as np
import sys
from Util import *

batch_size = 128
img_size = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
alpha = 1e-8
alpha_orig = 1e-8

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
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
vgg = torch.load('pretrain/vgglike_10.pk')
n_layer = 14
features_layerid = [0,3,7,10,14,17,20,24,27,30,34,37,40]
mask_list = []
for i in features_layerid:
  mask_list.append(torch.ones_like(vgg.features[i].bias))
mask_list.append(torch.ones(512,))
net = Model.VGGLIKE_neuron(vgg, mask_list, features_layerid)
net = net.to(device)

loss_ce = nn.CrossEntropyLoss()
alpha = 1e-9

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = loss_ce(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('forward train acc: ',correct/total, ' loss: ', forward_loss)
    return correct/total, forward_loss

clipper = ZeroOneClipper(n_layer)
rounder = RoundClipper([0,1,2,3,4,5,13], [], n_layer)

def train_back(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_prev = -10000
    for i in range(100000):
        inputs = torch.rand((batch_size, 3, img_size, img_size)).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, mask, alpha, batch_size)
        loss.backward(retain_graph=True)
        optimizer.step()
        net.apply(clipper)
        train_loss += loss.item()
        # print(abs(loss.item() - loss_prev))
        if abs(loss.item() - loss_prev) < 1e-2: break
        loss_prev = loss.item()
    print('backward train epoch: ',i)
    net.apply(rounder)
        
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, mask = net(inputs)
            loss = loss_ce(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('test acc: ', correct/total)
    return correct/total

targetacc = 0.90
print(net)

targets = [60,60,110,120,240,180,100,100,100,100,100,100,100,350]
for epoch in range(100000):
    optimizer = optim.Adam(net.vgg.parameters(), lr=lr/10)
    cnt = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
    forward_loss = 10000
    alpha = alpha*2 if alpha < alpha_orig*10 else alpha;
    while(test(epoch) < targetacc and cnt < 50 and forward_loss > 1e-5): 
      alpha = alpha_orig
      scheduler.step()
      forward_acc, forward_loss = train(epoch)
      cnt += 1

    optid = []
    for layerid in range(n_layer):
      if net.mask[layerid].data.nonzero().shape[0] < targets[layerid]: continue
      else: optid.append(layerid)
    print(optid)
    if len(optid) == 0: break

    optimizer = optim.Adam([net.mask[i] for i in optid], lr=1e-2)
    train_back(epoch)
    
    n_total = 0
    nonzero_total = 0
    for i in range(n_layer):
      n = net.mask[i].view(-1,).shape[0]
      nonzero = net.mask[i].data.nonzero().shape[0]
      n_total += n
      nonzero_total += nonzero
      print('layer ',i,' : ',float(n-nonzero)/n, ' ==> ', nonzero,'/',n)
         
#post train
torch.save(net, 'checkpoint/spnn_vgg.pk')
print("-------------- post train ------------")
optimizer = optim.Adam(net.vgg.parameters(), lr=lr/10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
for epoch in range(50):
    scheduler.step()
    train(epoch)
    test(epoch)
print("------------- sparsity -----------")
for i in range(n_layer):
  n = net.mask[i].view(-1,).shape[0]
  nonzero = net.mask[i].data.nonzero().shape[0]
  n_total += n
  nonzero_total += nonzero
  print('layer ',i,' : ',float(n-nonzero)/n, ' ==> ', nonzero,'/',n)
 
