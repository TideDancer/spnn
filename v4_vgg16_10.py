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
import numpy as np
import sys
from Util import *
from copy import deepcopy

batch_size = 128
img_size = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print('forward train acc: ',correct/total, ' and loss: ', train_loss)
    return correct/total, train_loss

clamper = ZeroOneClipper(n_layer)

def train_back(epoch):
    net.train()
    correct = 0
    total = 0
    loss_prev = -10000
    for i in range(100000):
        inputs = torch.ones((batch_size, 3, img_size, img_size)).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, mask, alpha, batch_size)
        loss.backward(retain_graph=True)
        net.apply(clamper)
        optimizer.step()
        # if i % 1000 == 0 and i > 0: print(abs(loss.item() - loss_prev))
        if abs(loss.item() - loss_prev) < 1e-3: break
        loss_prev = loss.item()
    print('backward train epoch: ',i)
    net.apply(tailer)
        
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

targetacc = 0.9
print(net)

nclip_list = [1,1,2,2,4,4,4,16,16,16,16,16,16,8]
inc_list = [1,1,2,2,4,4,4,16,16,16,16,16,16,8]
# targets = [400, 250, 40]
reverse_list = [0]*n_layer
T = 5
for epoch in range(100000):
    for layerid in range(n_layer):
        # if net.mask[layerid].data.nonzero().shape[0] < targets[layerid]: break
        if reverse_list[layerid] > 0:
          reverse_list[layerid] -= 1
          print('***** skip layer ', layerid)
          print(reverse_list)
          continue
        print('optimize layer ', layerid)
        if 0 not in reverse_list and epoch > 50: T = T + 5 if T < 21 else T
        mask_tmp = deepcopy(net.mask[layerid].data)

        nclip_list[layerid] += inc_list[layerid]
        tailer = TailClipper(nclip_list, n_layer)
        optimizer = optim.Adam([net.mask[layerid]], lr=1e-2)
        train_back(epoch)

        optimizer = optim.Adam(net.vgg.parameters(), lr=lr/10)
        cnt = 0
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4, 7, 10, 20], gamma=0.5)
        forward_loss = 1000
        test_acc = test(epoch)
        while(test_acc < targetacc and cnt < T and forward_loss > 1e-5): 
          scheduler.step()
          forward_acc, forward_loss = train(epoch)
          test_acc = test(epoch)
          cnt += 1
        if test_acc < targetacc: 
          print('********** reverse layer ', layerid, ' *********')
          net.mask[layerid].data = mask_tmp
          nclip_list[layerid] -= inc_list[layerid]
          reverse_list[layerid] = 5
        
        n_total = 0
        nonzero_total = 0
        for i in range(n_layer):
          n = net.mask[i].view(-1,).shape[0]
          nonzero = net.mask[i].data.nonzero().shape[0]
          n_total += n
          nonzero_total += nonzero
          print('layer ',i,' : ',float(n-nonzero)/n, ' ==> ', nonzero,'/',n)
         
#post train
torch.save(net, 'checkpoint/v4_vgg_10.pk')
print("-------------- post train ------------")
optimizer = optim.Adam(net.vgg.parameters(), lr=lr/10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
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
 
