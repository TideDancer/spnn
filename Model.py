import torch
from torch import nn
import vgg
import torch.nn.functional as F

class LENET31(nn.Module):
    def __init__(self, _img_size, _out_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(_img_size*_img_size, 300), nn.ReLU(), nn.Linear(300,100), nn.ReLU(), nn.Linear(100, _out_size))
        self.input_size = _img_size
        self.output_size = _out_size
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return F.softmax(x)

class LENET31_neuron(nn.Module):
    def __init__(self, _lenet31, _mask_list, _mask_layerid):
        super().__init__()
        self.lenet = _lenet31
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_layerid = _mask_layerid
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x *= self.mask[0]
        k = 1
        for i in range(len(self.lenet.fc)):
          x = self.lenet.fc[i](x)
          if i in self.mask_layerid: 
            x *= self.mask[k]
            k += 1
        return x, self.mask

class LENET31_weight(nn.Module):
    def __init__(self, _lenet31, _mask_list, _mask_layerid):
        super().__init__()
        self.lenet = _lenet31
        self.mask = nn.ParameterList(list(map(lambda e: nn.Parameter(e), _mask_list)))
        self.mask_layerid = _mask_layerid
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        k = 0
        for i in range(len(self.lenet.fc)):
          if i in self.mask_layerid: 
            x = F.linear(x, self.lenet.fc[i].weight*self.mask[k], self.lenet.fc[i].bias)
            k += 1
          else:
            x = self.lenet.fc[i](x)
        return x, self.mask

class LENET53(nn.Module):
    def __init__(self, _img_size, _out_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(_img_size*_img_size, 500), nn.ReLU(), nn.Linear(500,300), nn.ReLU(), nn.Linear(300, _out_size))
        self.input_size = _img_size
        self.output_size = _out_size
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return F.softmax(x)

class LENET53_neuron(nn.Module):
    def __init__(self, _lenet53, _mask_list, _mask_layerid):
        super().__init__()
        self.lenet = _lenet53
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_layerid = _mask_layerid
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x *= self.mask[0]
        k = 1
        for i in range(len(self.lenet.fc)):
          x = self.lenet.fc[i](x)
          if i in self.mask_layerid: 
            x *= self.mask[k]
            k += 1
        return x, self.mask

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

class LENET5_weight(nn.Module):
    def __init__(self, _lenet5, _mask_list, _mask_featuresid, _mask_classifierid):
        super().__init__()
        self.lenet = _lenet5
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_featuresid = _mask_featuresid
        self.mask_classifierid = _mask_classifierid
    def forward(self, x):
        k = 0
        for i in range(len(self.lenet.features)):
          if i in self.mask_featuresid: 
            x = F.conv2d(self.lenet.features[i].weight*self.mask[k], self.lenet.features[i].bias)
            k += 1
          else:
            x = self.lenet.features[i](x)
        x = x.view(x.shape[0], -1)
        for i in range(len(self.lenet.classifier)):
          if i in self.mask_classifierid: 
            x = F.linear(x, self.lenet.classifier[i].weight*self.mask[k], self.lenet.classifier[i].bias)
            k += 1
          else:
            x = self.lenet.classifier[i](x)
        return x, self.mask

class LENET5_neuron(nn.Module):
    def __init__(self, _lenet5, _mask_list, _mask_featuresid, _mask_classifierid):
        super().__init__()
        self.lenet = _lenet5
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_featuresid = _mask_featuresid
        self.mask_classifierid = _mask_classifierid
    def forward(self, x):
        k = 0
        for i in range(len(self.lenet.features)):
          x = self.lenet.features[i](x)
          if i in self.mask_featuresid: 
            x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
            k += 1
        x = x.view(x.shape[0], -1)
        x *= self.mask[k]
        k += 1
        for i in range(len(self.lenet.classifier)):
          x = self.lenet.classifier[i](x)
          if i in self.mask_classifierid: 
            x *= self.mask[k]
            k += 1
        return x, self.mask

class VGGLIKE_neuron(nn.Module):
    def __init__(self, _vgg, _mask_list, _mask_featuresid):#, _mask_classifierid):
        super().__init__()
        self.vgg = _vgg
        self.mask = nn.ParameterList(map(lambda e: nn.Parameter(e), _mask_list)) 
        self.mask_featuresid = _mask_featuresid
        # self.mask_classifierid = _mask_classifierid
    def forward(self, x):
        k = 0
        for i in range(len(self.vgg.features)):
          x = self.vgg.features[i](x)
          if i in self.mask_featuresid: 
            x *= self.mask[k].view(-1,1).expand(-1,x.shape[2]*x.shape[3]).view(-1,x.shape[2],x.shape[3])
            k += 1
        x = x.view(x.shape[0], -1)
        x *= self.mask[k]
        k += 1
        x = self.vgg.classifier(x)
        # x *= self.mask[k]
        # k += 1
        return x, self.mask

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
