# config file
import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import Model

parser = argparse.ArgumentParser(description='SPNN')
parser.add_argument('arch', metavar='ARCH',
                    help='model architecture: lenet5,vgg16,resnet20')
parser.add_argument('dataset', metavar='DATASET',
                    help='dataset name: MNIST,CIFAR10')
parser.add_argument('path_data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('path_model', metavar='PRETRAINED',
                    help='pretrained model file')
parser.add_argument('-p', '--path_check', default='checkpoint/z.pk', metavar='CHECK',
                    help='path to checkpoint')
parser.add_argument('--save_format', default='net', metavar='FORMAT',
                    help='save format, net,state_dict')
parser.add_argument('target_acc', metavar='TARGETACC', type=float,
                    help='target acc: for comparison, suggest: vgg16(0.911), lenet5(0.9911), resnet(0.909)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-l', '--lr_forward', default=1e-3, type=float,
                    metavar='LRF', help='forward learning rate, default=1e-3 (suggest: resnet 1e-4')
parser.add_argument('--lr_adv', default=1e-2, type=float,
                    metavar='LRA', help='adv learning rate, default=1e-2')
parser.add_argument('-e', '--adv_eps', default=0.1, type=float,
                    metavar='EPS', help='adv bound, default=0.1 (suggest: vgg 5e-4)')
parser.add_argument('-s', '--stop_diff', default=1, type=float,
                    metavar='SD', help='adv training stopping if diff < SD, default=1')
parser.add_argument('--thres', default=0.9, type=float,
                    metavar='TH', help='clip elements < adv_eps*thres, default: 0.9')
parser.add_argument('-T', '--epoch', default=10, type=int,
                    metavar='T', help='forward training epoch, default=10')
parser.add_argument('-I', '--interval', default=5, type=int,
                    metavar='IT', help='learning rate decay (0.5) interval, default=5')
parser.add_argument('-r', '--path_res', default='', metavar='RESUME', 
                    help='resume from checkpoint file, if set will resume training')
parser.add_argument('--res_layer', default=0, type=int, metavar='RESUME', 
                    help='resume from which layer, only useful when resuming mode')
parser.add_argument('-E', default=100, type=int, metavar='EPOCH', 
                    help='epoch number')

args = parser.parse_args()

# ------------ dataset -----------
if args.dataset == 'CIFAR10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_size = 32
    n_channel = 3
    
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
    
    trainset = torchvision.datasets.CIFAR10(root=args.path_data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.path_data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)
    advloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
elif args.dataset == 'MNIST':
    img_size = 28
    n_channel = 1
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=False, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    advloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)

elif args.dataset == 'ImageNet':
    img_size = 224
    n_channel = 3
    traindir = args.path_data + 'train'
    valdir = args.path_data + 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True
            )
    testloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
            )
    advloader = trainloader


# ------------ net arch ----------
if args.arch=='alexnet' or args.arch=='resnet34' or args.arch=='resnet50' or args.arch=='resnet101':
    net = eval('torchvision.models.'+args.arch+'(pretrained=True)')
else:
    net = torch.load(args.path_model)

if args.arch in ['vgg16']:
    features_layerid = [0,3,7,10,14,17,20,24,27,30,34,37,40]
    classifier_layerid = []
    safe = [32,0,0,0,0,0,0,0,0,0,0,0,0]
elif args.arch == 'lenet5':
    features_layerid = [0,3]
    classifier_layerid = [0]
    safe = [4,10,100,0]
elif args.arch == 'lenet53':
    features_layerid = []
    classifier_layerid = [0,2]
    safe = [200, 160,50]
elif args.arch == 'alexnet':
    features_layerid = [0,3,6,8,10]
    classifier_layerid = [1,4]

# build masks
mask_list = []
if args.arch == 'resnet20':
    for layerid in range(1,4):
        for sublayerid in range(3):
            for convid in range(1,3):
                exec('mask_list.append(torch.ones(net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'.out_channels))')
elif args.arch=='resnet34':
    for layerid in range(1,5):
        for sublayerid in range(len(eval('net.layer'+str(layerid)))):
            for convid in range(1,3):
                exec('mask_list.append(torch.ones(net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'.out_channels))')
elif args.arch=='resnet50' or args.arch=='resnet101':
    for layerid in range(1,5):
        for sublayerid in range(len(eval('net.layer'+str(layerid)))):
            for convid in range(1,4):
                exec('mask_list.append(torch.ones(net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'.out_channels))')
else:
    for i in features_layerid:
        mask_list.append(torch.ones_like(net.features[i].bias))
    if args.arch == 'alexnet':
        mask_list.append(torch.ones(net.classifier[1].in_features))
    else:
        mask_list.append(torch.ones(net.classifier[0].in_features))
    for i in classifier_layerid:
        mask_list.append(torch.ones_like(net.classifier[i].bias))
n_layer = len(mask_list)
    
# build net with mask
if args.path_res == '':
    if args.arch == 'lenet53':
        net = Model.FC_Mask(net, mask_list, classifier_layerid)
    elif args.arch == 'resnet20':
        net = Model.RESNET20_Mask(net, mask_list)
    elif args.arch=='resnet34':
        net = Model.RESNET_BASIC_Mask(net, mask_list)
    elif args.arch=='resnet50' or args.arch=='resnet101':
        net = Model.RESNET_BOTTLENECK_Mask(net, mask_list)
    else:
        net = Model.CONV_Mask(net, mask_list, features_layerid, classifier_layerid)
    
    if args.arch=='alexnet' or args.arch=='resnet34' or args.arch=='resnet50' or args.arch=='resnet101':
        net = torch.nn.DataParallel(net.cuda())
    else: 
        net = net.cuda()
else:
    net = torch.load(args.path_res)

print(net)


# ------------ others ----------
loss_ce = nn.CrossEntropyLoss()
alpha = 1e-2
ITER_NUM = 10000
LARGE_NUM = 10000
SMALL_NUM = 1e-3

cnt = 0
interval_seq = []
while(cnt<args.epoch):
    interval_seq.append(cnt)
    cnt += args.interval
 
# res34
if args.arch=='resnet34':
    netlayers ={0:  'layer1.0', 
            1:  'layer1.0', 
            2:  'layer1.1', 
            3:  'layer1.1', 
            4:  'layer1.2',
            5:  'layer1.2',

            6:  'layer2.0',
            7:  'layer2.0',
            8:  'layer2.1',
            9:  'layer2.1',
            10: 'layer2.2',
            11: 'layer2.2',
            12: 'layer2.3',
            13: 'layer2.3',

            14: 'layer3.0',
            15: 'layer3.0',
            16: 'layer3.1',
            17: 'layer3.1',
            18: 'layer3.2',
            19: 'layer3.2',
            20: 'layer3.3',
            21: 'layer3.3',
            22: 'layer3.4',
            23: 'layer3.4',
            24: 'layer3.5',
            25: 'layer3.5',

            26: 'layer4.0', 
            27: 'layer4.0', 
            28: 'layer4.1', 
            29: 'layer4.1', 
            30: 'layer4.2',
            31: 'layer4.2',}

if args.arch=='resnet50':
    netlayers ={0:  'layer1[0].parameters()',
            1:  'layer1[0].parameters()',
            2:  'layer1[0].parameters()',
            3:  'layer1[1].parameters()',
            4:  'layer1[1].parameters()',
            5:  'layer1[1].parameters()',
            6:  'layer1[2].parameters()',
            7:  'layer1[2].parameters()',
            8:  'layer1[2].parameters()',

            9:  'layer2[0].parameters()',
            10: 'layer2[0].parameters()',
            11: 'layer2[0].parameters()',
            12: 'layer2[1].parameters()',
            13: 'layer2[1].parameters()',
            14: 'layer2[1].parameters()',
            15: 'layer2[2].parameters()',
            16: 'layer2[2].parameters()',
            17: 'layer2[2].parameters()',
            18: 'layer2[3].parameters()',
            19: 'layer2[3].parameters()',
            20: 'layer2[3].parameters()',

            21: 'layer3[0].parameters()',
            22: 'layer3[0].parameters()',
            23: 'layer3[0].parameters()',
            24: 'layer3[1].parameters()',
            25: 'layer3[1].parameters()',
            26: 'layer3[1].parameters()',
            27: 'layer3[2].parameters()',
            28: 'layer3[2].parameters()',
            29: 'layer3[2].parameters()',
            30: 'layer3[3].parameters()',
            31: 'layer3[3].parameters()',
            32: 'layer3[3].parameters()',
            33: 'layer3[4].parameters()',
            34: 'layer3[4].parameters()',
            35: 'layer3[4].parameters()',
            36: 'layer3[5].parameters()',
            37: 'layer3[5].parameters()',
            38: 'layer3[5].parameters()',

            39: 'layer3[0].parameters()',
            40: 'layer3[0].parameters()',
            41: 'layer3[0].parameters()',
            42: 'layer3[1].parameters()',
            43: 'layer3[1].parameters()',
            44: 'layer3[1].parameters()',
            45: 'layer3[2].parameters()',
            46: 'layer3[2].parameters()',
            47: 'layer3[2].parameters()',
}
