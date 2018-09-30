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

batch_size = 32
img_size = 28
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

# Model
# lenet31 = Model.LENET31(28, 10)
lenet31 = torch.load('pretrain/lenet31.pk')
n_layer = 3
classifier_layerid = [0,2]
# mask_list = []
mask_list = [torch.ones((img_size*img_size),)]
for i in classifier_layerid:
#  mask_list.append(torch.ones_like(lenet31.fc[i].weight))
  mask_list.append(torch.ones_like(lenet31.fc[i].bias))
# net = Model.LENET31_weight(lenet31, mask_list, classifier_layerid)
net = Model.LENET31_neuron(lenet31, mask_list, classifier_layerid)
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
        inputs = torch.ones((batch_size, 1, img_size, img_size)).to(device)
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

targetacc = 0.98
print(net)

nclip_list = [400, 100, 50]
inc_list = [1, 1, 1]
# targets = [400, 250, 40]
reverse_list = [0,0,0]
T = 10

optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
cnt = 0
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
forward_loss = 1000
for layerid in range(n_layer):
  tailer = TailClipper(nclip_list, n_layer)
  optimizer = optim.Adam([net.mask[layerid]], lr=1e-2)
  train_back(cnt)
  test_acc = test(cnt)
  while(test_acc < 0.96): 
    scheduler.step()
    forward_acc, forward_loss = train(cnt)
    test_acc = test(cnt)
    cnt += 1
 
while(test_acc < targetacc and cnt < 60 and forward_loss > 1e-7): 
  scheduler.step()
  forward_acc, forward_loss = train(cnt)
  test_acc = test(cnt)
  cnt += 1
 
for epoch in range(100000):
    for layerid in range(n_layer):
        # if net.mask[layerid].data.nonzero().shape[0] < targets[layerid]: break
        if reverse_list[layerid] > 0:
          reverse_list[layerid] -= 1
          print('***** skip layer ', layerid)
          print(reverse_list)
          continue
        if epoch > 100 and 0 not in reverse_list: T = T+10 if T < 41 else T
        print('optimize layer ', layerid)
        mask_tmp = deepcopy(net.mask[layerid].data)

        nclip_list[layerid] += inc_list[layerid]
        tailer = TailClipper(nclip_list, n_layer)
        optimizer = optim.Adam([net.mask[layerid]], lr=1e-2)
        train_back(epoch)

        optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
        cnt = 0
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8, 15, 30], gamma=0.5)
        forward_loss = 1000
        test_acc = test(epoch)
        while(test_acc < targetacc and cnt < T and forward_loss > 1e-7): 
          scheduler.step()
          forward_acc, forward_loss = train(epoch)
          test_acc = test(epoch)
          cnt += 1
        if test_acc < targetacc: 
          print('********** reverse layer ', layerid, ' *********')
          net.mask[layerid].data = mask_tmp
          nclip_list[layerid] -= inc_list[layerid]
          reverse_list[layerid] = 10
        
        n_total = 0
        nonzero_total = 0
        for i in range(n_layer):
          n = net.mask[i].view(-1,).shape[0]
          nonzero = net.mask[i].data.nonzero().shape[0]
          n_total += n
          nonzero_total += nonzero
          print('layer ',i,' : ',float(n-nonzero)/n, ' ==> ', nonzero,'/',n)
         
#post train
torch.save(net, 'checkpoint/spnn_lenet31.pk')
print("-------------- post train ------------")
optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
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
 
