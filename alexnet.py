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
import ModelAlexnet

batch_size = 4
img_size = 32
lr = 1e-4
alpha = float(sys.argv[1])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
alexnet = ModelAlexnet.alexnet(pretrained=True)
alexnet = alexnet.to(device)
conv_layerid = [0, 3, 6, 8, 10]
linear_layerid = [1, 4, 6]
mask_list = []
for i in conv_layerid:
    mask_list.append(torch.ones_like(alexnet.features[i].weight))
for i in linear_layerid:
    mask_list.append(torch.ones_like(alexnet.classifier[i].weight))
n_feature = 13
n_classifier = 7
net = Model.ALEXNET_mask(alexnet, mask_list)
# print(net)
# for name, param in net.named_parameters():
#     if param.requires_grad:
#       print(name, param.data.shape)
# sys.exit()

loss_ce = nn.CrossEntropyLoss()
def loss_mask(prediction, label, mask):
  loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
  loss_out = -torch.sum(torch.sum(prediction))
  return alpha*loss_l1 + loss_out

def print_sparsity(n_layer, mask_list):
    nonzero_total = 0
    n_total = 0
    for i in range(n_layer):
      n = mask_list[i].data.view(-1).shape[0]
      nonzero = mask_list[i].data.nonzero().shape[0]
      n_total += n
      nonzero_total += nonzero
      #print(float(n-nonzero)/n)
    print(float(n_total/nonzero_total)) 

# Training
def train(epoch, z):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if z > 0: 
          if batch_idx > len(trainloader)/z: break
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
        if batch_idx % 20 == 0: print(batch_idx, '/', len(trainloader), ' ==> forward train acc: ',correct/total)
        # print_sparsity(n_layer, mask_list)
    print('forward train acc: ',correct/total)
    torch.save(net, '/opt/ml/disk/spnn/net.pk')

class ZeroOneClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        # filter the variables to get the ones you want
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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
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

# pretrain
trainloader = torch.utils.data.DataLoader(datasets.ImageFolder('/opt/ml/disk/data/imagenet/'), batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(datasets.ImageFolder('/opt/ml/disk/data/imagenet/'), batch_size=batch_size, shuffle=True, num_workers=8)
optimizer = optim.Adam(net.alexnet.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 2000, 3000], gamma=0.1)

for epoch in range(100):
    scheduler.step()
    train(epoch, 0)
    if epoch % 1 == 0: test(epoch)
sys.exit()

for epoch in range(500):
    trainloader = torch.utils.data.DataLoader(datasets.CIFAR10('/opt/ml/disk/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(datasets.CIFAR10('/opt/ml/disk/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)

    net = Model.VGGLIKE_mask(conv_layerid, linear_layerid, n_feature, n_classifier, vgg16, mask_list)
    net = net.to(device)
    optimizer = optim.Adam(net.vgg.parameters(), lr=lr)
    train(epoch, step1)

    net = Model.VGGLIKE_mask(conv_layerid, linear_layerid, n_feature, n_classifier, vgg16, mask_list)
    net = net.to(device)
    optimizer = optim.Adam(net.mask.parameters(), lr=lr*10)
    mask_list = train_back(epoch, step2)
    
    if epoch % 10 == 0:
      test(epoch)
      print_sparsity(n_layer, mask_list)

#post train
for epoch in range(100):
    net = Model.VGGLIKE_mask(conv_layerid, linear_layerid, n_feature, n_classifier, vgg16, mask_list)
    net = net.to(device)
    optimizer = optim.Adam(net.vgg.parameters(), lr=lr/10)
    train(epoch, 0)


