import torch
import torchvision
import sys
import numpy

def count_parameters(model):
    total_param = 0
    paramnum_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = numpy.prod(param.size())
            if 'shortcut' in name or 'downsample' in name or 'fc' in name or 'classifier.2.weight' in name: 
                total_param += num_param
                continue
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                # print(param.size())
                paramnum_list.append(num_param)
            # else:
                # print(name, ':', num_param)
            total_param += num_param
    return total_param, paramnum_list

print('python3 calculate_params.py name model orig layer')
name = sys.argv[1]
if name == 'resnet34':
    net = torchvision.models.resnet34(pretrained=True)
if name == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)
elif name == 'resnet101':
    net = torchvision.models.resnet101(pretrained=True)
else:
    net = torch.load(sys.argv[2])

if name == 'resnet34':
    orig = [64,64, 64,64, 64,64, 128,128, 128,128, 128,128, 128,128, 256,256, 256,256,  256,256, 256,256, 256,256, 256,256,  512,512, 512,512, 512,512]
    # orig = [1, 64,64, 64,64, 64,64, 128,128, 1, 128,128, 128,128, 128,128, 256,256, 1, 256,256,  256,256, 256,256, 256,256, 256,256,  512,512, 1, 512,512, 512,512, 1]
    # layers = [1, 46,46, 48,48, 47,47, 105,105, 1, 101,101, 93,93, 94,94,   213,213, 1, 202,202,  202,202, 200,200, 198,198, 177,177,  381,381, 1, 360,360, 410,410, 1]
    # layers = [44,58,54,54,54,55,108,104,102,104,99,111,107,108,213,231,217,210,217,197,226,214,226,209,230,215,406,439,446,424,427,474]
elif name == 'resnet50':
    orig = [64,64,256,64,64,256,64,64,256,128,128,512,128,128,512,128,128,512,128,128,512,256,256,1024,256,256,1024,256,256,1024,256,256,1024,256,256,1024,256,256,1024,512,512,2048,512,512,2048,512,512,2048]
elif name == 'resnet101':
    orig = []
# layers = [166,168,205,354,385,405,381,706,747,771,781,778,794,789,817,824,853,878,874,899,887,915,907,901,905,912,914,921,912,916,1700,1807,1983]
# print(len(layers))
elif name == 'resnet20':
    orig = [16,16,16,16,16,16,32,32,32,32,32,32,64,64,64,64,64,64]
# layers = [1,1,1,1,12,11,6,19,25,21,1,1,57,56,53,46,51,24]
elif name == 'vgg15':
    orig = [96,96,192,192,384,384,384,768,768,768,768,768,768,768]
elif name == 'vgg10':
    orig = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
elif name == 'lenet5':
    orig = [20,50,800,500]

layers = eval(sys.argv[3])

# net = torch.load(sys.argv[1])
# orig = eval(sys.argv[2])
# layers = eval(sys.argv[3])
t, l = count_parameters(net)
if name == 'resnet34' or name=='resnet50': l.pop(0)
if len(l)!=len(orig):
  print('wrong')
  sys.exit()
necessary = t - sum(l)
print('# layers: ',len(l),', layers: ', l)
print('t: ', t, ', sum of l: ',sum(l), ', neces: ', necessary)

nums = []
for i in range(len(orig)):
  nums.append(layers[i]/orig[i]*l[i])

# print(nums)
print('compression: ',t/(sum(nums)+necessary))
print('sparsity: ',(sum(nums)+necessary)/t)

