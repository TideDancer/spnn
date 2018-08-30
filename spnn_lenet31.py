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
def loss_mask(prediction, mask):
  alpha = 1e-8
  loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
  # loss_out = -sum([prediction[i,label[i]] for i in range(label.shape[0])])
  loss_out = -sum(sum(prediction))/batch_size
  # loss_out = -sum(prediction[:,0])
  # return alpha*loss_l1 + loss_out
  return loss_out

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
    # torch.save(net, 'model/lenet31weight.pk')

class ZeroOneClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(n_layer):
            w = module.mask[i].data
            w.clamp_(0, 1)
clipper = ZeroOneClipper()

def train_back(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for i in range(100000):
        inputs = torch.rand((batch_size, 1, img_size, img_size)).to(device)
        optimizer.zero_grad()
        outputs, mask = net(inputs)
        loss = loss_mask(outputs, mask)
        loss.backward(retain_graph=True)
        # print(net.mask[0].grad, torch.sum(torch.abs(net.mask[0].grad)))
        # for name, param in net.named_parameters():
        #     # if param.requires_grad and 'mask' in name:
        #       print(name, torch.sum(torch.abs(param.grad)))
        # sys.exit()
        optimizer.step()
        net.apply(clipper)
        train_loss += loss.item()
        if loss.item() < 1e-2: break
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

trainloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True)
targetacc = 0.97
print(net)

targets = [278, 86, 13]
for layerid in range(n_layer):
    if os.path.isfile('checkpoint/spnn_lenet31_neuron_'+str(layerid)+'.pk'):
        net = torch.load('checkpoint/spnn_lenet31_neuron_'+str(layerid)+'.pk')
        continue
    print("============> optimize layer ", layerid, "<============")
    for epoch in range(10000):
        optimizer = optim.Adam(net.lenet.parameters(), lr=lr)
        cnt = 0
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.5)
        while(test(epoch) < targetacc and cnt < 150): 
          scheduler.step()
          train(epoch)
          cnt += 1
    
        optimizer = optim.Adam([net.mask[layerid]], lr=1e-3)
        back_loss_previous = train_back(epoch)
        
        # if epoch % 1 == 0:
        n_total = 0
        nonzero_total = 0
        for i in range(n_layer):
          n = net.mask[i].view(-1,).shape[0]
          nonzero = net.mask[i].data.nonzero().shape[0]
          n_total += n
          nonzero_total += nonzero
          print('layer ',i,' : ',float(n-nonzero)/n, ' ==> ', nonzero,'/',n)
        # print('compression: ',float(n_total/nonzero_total)) 
         
        if net.mask[layerid].data.nonzero().shape[0] < targets[layerid]: 
          torch.save(net, 'checkpoint/spnn_lenet31_neuron_'+str(layerid)+'.pk')
          break

#post train
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.5)
optimizer = optim.Adam(net.lenet.parameters(), lr=lr/10)
for epoch in range(150):
    scheduler.step()
    train(epoch)
    test(epoch)


