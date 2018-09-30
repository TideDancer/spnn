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

batch_size = 8
img_size = 28
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
alpha = 1e-9
alpha_orig = 1e-9

trainloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

# Model
lenet53 = torch.load('pretrain/lenet53.pk')
n_layer = 3
classifier_layerid = [0,2]
mask_list = [torch.ones((img_size*img_size),)]
for i in classifier_layerid:
  mask_list.append(torch.ones_like(lenet53.fc[i].bias))
net = Model.LENET53_neuron(lenet53, mask_list, classifier_layerid)
net = net.to(device)

loss_ce = nn.CrossEntropyLoss()
def loss_mask(prediction, mask):
  loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
  loss_out = -sum(sum(prediction))/batch_size
  return alpha*loss_l1 + loss_out

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
    print('forward train acc: ',correct/total)
    return correct/total

class ZeroOneClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(n_layer):
            w = module.mask[i].data
            w.clamp_(0, 1)
clipper = ZeroOneClipper()

class RoundClipper(object):
    def __init__(self, ceillist, floorlist):
        self.ceillist = ceillist
        self.floorlist = floorlist
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(n_layer):
            w = module.mask[i].data
            if i in self.ceillist: w.ceil_()
            elif i in self.floorlist: w.floor_()
            else: w.round_()
rounder = RoundClipper([1,2], [])

def train_back(epoch):
    net.train()
    correct = 0
    total = 0
    loss_prev = -10000
    for i in range(100000):
        inputs = torch.ones((batch_size, 1, img_size, img_size)).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, mask)
        loss.backward(retain_graph=True)
        optimizer.step()
        net.apply(clipper)
        if abs(loss.item() - loss_prev) < 1e-3: break
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

targetacc = 0.98
print(net)

targets = [250, 170, 60]
for epoch in range(100000):
    optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
    cnt = 0
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    forward_acc = 0
    alpha = alpha*2 if alpha < alpha_orig*50 else alpha;
    while(test(epoch) < targetacc and cnt < 40 and forward_acc < 1.0): 
      alpha = alpha_orig
      scheduler.step()
      forward_acc = train(epoch)
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
torch.save(net, 'checkpoint/spnn_lenet53.pk')
print("-------------- post train ------------")
optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
for epoch in range(100):
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
 

