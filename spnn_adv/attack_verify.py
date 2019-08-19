import foolbox
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import argparse
import sys
import os
import pickle
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> device: ', device)

# cifar10 dataset
if sys.argv[1] == 'CIFAR10':
    n_class = 10
    n_channel = 3
    img_size = 32
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=6)
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=6)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

# # mnist dataset
if sys.argv[1] == 'MNIST':
    n_class = 10
    n_channel = 1
    img_size = 28
    
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/data', train=True, download=True,
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=1, shuffle=False, num_workers=6 )
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=False, num_workers=6 )
    
    mean = np.array([0.1307]).reshape((1, 1, 1))
    std = np.array([0.3081]).reshape((1, 1, 1))


# instantiate the model
print('==> start model ...')
model = torch.load(sys.argv[2], map_location=device)
model = model.to(device) #.eval()
print(model)
try:
    for i in range(14):
        print(i, torch.sum(model.mask[i]))
    print(model.mask[0])
except:
    print('==> no mask')
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=n_class, preprocessing=(mean, std))

# set attack method
distance = 'Linfinity'
if   sys.argv[3] == 'FGSM':   attack_str = 'foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.'+distance+')'
elif sys.argv[3] == 'PGD':  attack_str = 'foolbox.attacks.PGD(fmodel, distance=foolbox.distances.'+distance+')'
elif sys.argv[3] == 'CWL2': attack_str = 'foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=foolbox.distances.MeanSquaredDistance)'
elif sys.argv[3] == 'DeepFool':     attack_str = 'foolbox.attacks.DeepFoolAttack(fmodel, distance=foolbox.distances.'+distance+')'
elif sys.argv[3] == 'BIML2':attack_str = 'foolbox.attacks.L2BasicIterativeAttack(fmodel, distance=foolbox.distances.MeanSquaredDistance)'
elif sys.argv[3] == 'BIML1':attack_str = 'foolbox.attacks.L1BasicIterativeAttack(fmodel, distance=foolbox.distances.MeanAbsoluteDistance)'
else:                       attack_str = 'foolbox.attacks.PGD(fmodel, distance=foolbox.distances.Linfinity)'   

# others
pred_success = 0
adv_success = 0

import matplotlib.pyplot as plt
import torch.optim as optim
import copy
attack_bound = 1e-2

def create_one_sample(image, label):
    attack = eval(attack_str)
    image = torch.reshape(image, (n_channel,img_size,img_size)).data.numpy()
    label = label.numpy()[0]
    # perform attack 
    if sys.argv[3]=='PGD' or 'BIM' in sys.argv[3]:
        adversarial = attack(image, label=label, unpack=False, binary_search=False, epsilon=attack_bound)
    else:
        adversarial = attack(image, label=label, unpack=False) 

    if adversarial.distance.value == 0 or adversarial.distance.value == np.inf: # orig pred is wrong, return orig image / or attack faile, return none image
        return 0 # both means input image to detector is original image
    else:
        return 1 # input image to detector is adv image

# train
def train(epoch):
    cnt = 0
    for batch_idx, (image, label) in enumerate(trainloader):
        try:
            cnt += create_one_sample(image, label)
        except:
            print('batch ', batch_idx, ' in train has error ')
        print(batch_idx, cnt)
    print('train epoch ', epoch, ' num-of-success-attack ', cnt)
    return cnt

def test(epoch):
    cnt = 0
    for batch_idx, (image, label) in enumerate(testloader):
        try:
            cnt += create_one_sample(image, label)
        except:
            print('batch ', batch_idx, ' in test has error ')
        print(batch_idx, cnt)
    print('test epoch ', epoch, ' num-of-success-attack ', cnt)
    return cnt

# main loop
for epoch in range(1):
    # cnt = train(epoch)
    cnt = test(epoch)

