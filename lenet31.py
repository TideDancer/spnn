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

batch_size = 128
img_size = 28
lr = 1e-4
alpha = float(sys.argv[1])
z = int(sys.argv[2])
backlr = float(sys.argv[3])
targetacc = float(sys.argv[4])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

# Model
n_layer = 3
fc0 = nn.Linear(img_size*img_size, 300)
fc1 = nn.Linear(300, 100)
fc2 = nn.Linear(100, 10)
linear_list = [fc0, fc1, fc2]
mask_list = []
for fc in linear_list:
  mask_list.append(torch.ones_like(fc.weight))

loss_ce = nn.CrossEntropyLoss()
def loss_mask(prediction, label, mask):
  loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
  loss_out = -sum([prediction[i,label[i]] for i in range(label.shape[0])])
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
    # torch.save(net, '/opt/ml/disk/spnn/lenet31net.pk')

class ZeroOneClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(n_layer):
            w = module.mask[i].data
            w.clamp_(0, 1)
clipper = ZeroOneClipper()

def train_back(epoch, z):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if z > 0: 
          if batch_idx > len(trainloader)/z: break
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs = torch.ones_like(inputs).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, targets, mask)
        loss.backward(retain_graph=True)
        optimizer.step()
        net.apply(clipper)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('backward train loss: ',train_loss)
    return train_loss
        
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

trainloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
net = torch.load('/opt/ml/disk/spnn/lenet31net.pk')
net = net.to(device)
optimizer = optim.Adam(net.fc.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.1)

# pretrain
for epoch in range(500):
    scheduler.step()
    train(epoch)
    test(epoch)
sys.exit()

for epoch in range(2000):
    trainloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(net.fc.parameters(), lr=lr)
    while(test(epoch) < targetacc): train(epoch)

    optimizer = optim.Adam(net.mask.parameters(), lr=backlr)
    back_loss_previous = train_back(epoch, z)
    
    if epoch % 1 == 0:
      n_total = 0
      nonzero_total = 0
      for i in range(n_layer-1):
        n = mask_list[i].view(-1,).shape[0]
        nonzero = mask_list[i].data.nonzero().shape[0]
        n_total += n
        nonzero_total += nonzero
        print('layer ',i,' : ',float(n-nonzero)/n)
      print('compression: ',float(n_total/nonzero_total)) 

#post train
for epoch in range(200):
    optimizer = optim.Adam(net.fc.parameters(), lr=lr/10)
    train(epoch, 0)
    if epoch % 20 == 0: test(epoch)


