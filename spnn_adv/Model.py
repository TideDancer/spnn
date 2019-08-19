import torch
from torch import nn
import vgg
import torch.nn.functional as F

class LENET31(nn.Module):
    def __init__(self, _img_size, _out_size):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(_img_size*_img_size, 300), nn.ReLU(), nn.Linear(300,100), nn.ReLU(), nn.Linear(100, _out_size))
        self.input_size = _img_size
        self.output_size = _out_size
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return F.softmax(x)

class LENET53(nn.Module):
    def __init__(self, _img_size, _out_size):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(_img_size*_img_size, 500), nn.ReLU(), nn.Linear(500,300), nn.ReLU(), nn.Linear(300, _out_size))
        self.input_size = _img_size
        self.output_size = _out_size
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return F.softmax(x)

class LENET5(nn.Module):
    def __init__(self, _img_size, _in_channel, _out_size):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(_in_channel,20,5), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(20,50,5), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Linear(800,500), nn.ReLU(), nn.Linear(500, _out_size))
        self.input_channel = _in_channel
        self.input_size = _img_size
        self.output_size = _out_size
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return F.softmax(x)

## Fully connected network with mask
class FC_Mask(nn.Module):
    def __init__(self, _net, _mask_list, _mask_layerid):
        super().__init__()
        self.net = _net
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_layerid = _mask_layerid
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x *= self.mask[0]
        k = 1
        for i in range(len(self.net.classifier)):
          x = self.net.classifier[i](x)
          if i in self.mask_layerid: 
            x *= self.mask[k]
            k += 1
        return x

## feature (conv layers) - classifier (fc layers) network with mask
class CONV_Mask(nn.Module):
    def __init__(self, _net, _mask_list, _mask_featuresid, _mask_classifierid):
        super().__init__()
        self.net = _net
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_featuresid = _mask_featuresid
        self.mask_classifierid = _mask_classifierid
    def forward(self, x):
        k = 0
        for i in range(len(self.net.features)):
          x = self.net.features[i](x)
          if i in self.mask_featuresid: 
            x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
            k += 1
        x = x.view(x.shape[0], -1)
        x *= self.mask[k]
        k += 1
        for i in range(len(self.net.classifier)):
          x = self.net.classifier[i](x)
          if i in self.mask_classifierid: 
            x *= self.mask[k]
            k += 1
        return x


# class RESNET20_Mask(nn.Module):
#     def __init__(self, _net, _mask_list):
#         super().__init__()
#         self.net = _net
#         self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
#     def forward(self, x):
#         x = F.relu(self.net.bn1(self.net.conv1(x)))
#         k = 0
#         for i in range(3):
#             x = self.net.layer1[i](x)
#             x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
#             k += 1
#         for i in range(3):
#             x = self.net.layer2[i](x)
#             x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
#             k += 1
#         for i in range(3):
#             x = self.net.layer3[i](x)
#             x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
#             k += 1
#         x = F.avg_pool2d(x, 4)
#         x = x.view(x.shape[0],-1)
#         x = self.net.linear(x)
#         return x

class RESNET20_Mask(nn.Module):
    def __init__(self, _net, _mask_list):
        super().__init__()
        self.net = _net
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
    def forward(self, x):
        x = F.relu(self.net.bn1(self.net.conv1(x)))
        k = 0
        for layerid in range(1,4):
            for sublayerid in range(3):
                x_shortcut = x
                for convid in range(1,3):
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'(x)')
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].bn'+str(convid)+'(x)')
                    x = x*self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
                    k += 1
                    if convid == 1: x = F.relu(x)
                x = x + eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].shortcut(x_shortcut)')
                x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.shape[0],-1)
        x = self.net.linear(x)
        return x

class RESNET_BASIC_Mask(nn.Module):
    def __init__(self, _net, _mask_list):
        super().__init__()
        self.net = _net
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        k = 0
        for layerid in range(1,5):
            for sublayerid in range(len(eval('self.net.layer'+str(layerid)))):
                x_shortcut = x
                for convid in range(1,3):
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'(x)')
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].bn'+str(convid)+'(x)')
                    x = x*self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
                    k += 1
                    if convid == 1: x = F.relu(x)
                if (layerid > 1) and (sublayerid == 0):
                    x = x + eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].downsample(x_shortcut)')
                else:
                    x = x + x_shortcut
                x = F.relu(x)
        x = self.net.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.net.fc(x)
        return x

class RESNET_BOTTLENECK_Mask(nn.Module):
    def __init__(self, _net, _mask_list):
        super().__init__()
        self.net = _net
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        k = 0
        for layerid in range(1,5):
            for sublayerid in range(len(eval('self.net.layer'+str(layerid)))):
                x_shortcut = x
                for convid in range(1,4):
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].conv'+str(convid)+'(x)')
                    x = eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].bn'+str(convid)+'(x)')
                    x = x*self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
                    k += 1
                    if convid < 3: x = F.relu(x)
                if (sublayerid == 0):
                    x = x + eval('self.net.layer'+str(layerid)+'['+str(sublayerid)+'].downsample(x_shortcut)')
                else:
                    x = x + x_shortcut
                x = F.relu(x)
        x = self.net.avgpool(x)
        x = x.view(x.shape[0],-1)
        x = self.net.fc(x)
        return x

class NET_Straight(nn.Module):
    def __init__(self, _net):
        super().__init__()
        self.net = _net
    def forward(self, x):
        return self.net(x)
 

def test():
    import vgg
    vgg16 = vgg.VGG('VGG16')
    conv_layerid = [0,3,7,10,14,17,20,24,27,30,34,37,40]
    linear_layerid = [0,3]
    mask_list = []
    for i in conv_layerid:
        mask_list.append(torch.ones_like(vgg16.features[i].weight))
    for i in linear_layerid:
        mask_list.append(torch.ones_like(vgg16.classifier[i].weight))
    n_feature = 45
    n_classifier = 4
    net = VGGLIKE_mask(conv_layerid, linear_layerid, n_feature, n_classifier, vgg16, mask_list)
    net = torch.nn.DataParallel(net)
    x = torch.randn(2,3,32,32)
    y, _ = net(x)
    for name, param in net.named_parameters():
        print(name, param.shape)
    print(y.shape)

# test() 
