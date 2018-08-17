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
# lenet5 = Model.LENET5_standard()
n_layer = 4
# mask_list = []
# mask_list.append(torch.ones_like(lenet5.conv1.weight))
# mask_list.append(torch.ones_like(lenet5.conv2.weight))
# mask_list.append(torch.ones_like(lenet5.fc1.weight))
# mask_list.append(torch.ones_like(lenet5.fc2.weight))
# net = Model.LENET5_mask(img_size, lenet5, mask_list)
# net = net.to(device)

net = torch.load('/opt/ml/disk/spnn/lenet5net.pk')
net = net.to(device)
optimizer = optim.Adam(net.lenet.parameters(), lr=lr)
# for name, param in net.named_parameters():
#   if param.requires_grad:
#     print(name, param.data.shape)
# sys.exit()

loss_ce = nn.CrossEntropyLoss()
def loss_mask(prediction, label, mask):
  loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
  loss_out = -torch.sum(torch.sum(prediction))
  # print(loss_l1, loss_out)
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
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('forward train acc: ',correct/total)
    torch.save(net, '/opt/ml/disk/spnn/lenet5net.pk')

class ZeroOneClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(n_layer):
            w = module.mask[i].data
            w.clamp_(0, 1)
            # w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w))
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
        inputs = torch.ones_like(inputs).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, targets, mask)
        loss.backward(retain_graph=True)
        optimizer.step()
        net.apply(clipper)
        train_loss += loss.item()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('backward train loss: ',train_loss/(batch_idx+1))
    return mask
        
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
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('test acc: ', correct/total)
    return correct/total

# pretrain
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.1)
for epoch in range(400):
    train(epoch)
    test(epoch)
sys.exit()

for epoch in range(2000):
    trainloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(datasets.MNIST('/opt/ml/disk/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    # scheduler.step()

    optimizer = optim.Adam(net.lenet.parameters(), lr=lr)
    while(test(epoch) < targetacc): train(epoch)

    optimizer = optim.Adam(net.mask.parameters(), lr=backlr)
    mask_list = train_back(epoch, z)

    if epoch % 5 == 0:
      n_total = 0
      nonzero_total = 0
      for i in range(n_layer):
        n = mask_list[i].data.view(-1).shape[0]
        nonzero = mask_list[i].data.nonzero().shape[0]
        n_total += n
        nonzero_total += nonzero
        print('leyer ',i,': ',float(n-nonzero)/n)
      print('compression: ',float(n_total/nonzero_total)) 

#post train
for epoch in range(100):
    optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
    train(epoch, 0)
    if epoch % 20 == 0 : test(epoch)

