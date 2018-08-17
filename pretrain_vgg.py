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
import vgg

batch_size = 128
img_size = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
net = vgg.VGG('VGG16')
# net = torch.load('pretrain/vgg16.pk')
net = net.to(device)

loss_ce = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_ce(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('forward train acc: ',correct/total)
       
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = loss_ce(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('test acc: ', correct/total)
    return correct/total

# pretrain
trainloader = torch.utils.data.DataLoader(datasets.CIFAR10('~/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.CIFAR10('~/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(net.vgg.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.5)

for epoch in range(150):
    scheduler.step()
    train(epoch)
    test(epoch)
    if epoch % 30 == 0: 
        torch.save(net, 'pretrain/vgg16.pk')
        print('epoch: ', epoch)

