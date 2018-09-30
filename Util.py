import torch

def loss_mask(prediction, mask, alpha, batch_size):
    loss_l1 = sum(list(map(lambda e: torch.sum(torch.abs(e)), mask)))
    loss_out = -sum(sum(prediction))/batch_size
    return alpha*loss_l1 + loss_out
    # return loss_out

class ZeroOneClipper(object):
    def __init__(self, n_layer):
        self.n_layer = n_layer
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(self.n_layer):
            w = module.mask[i].data
            w.clamp_(0, 1)

class RoundClipper(object):
    def __init__(self, ceillist, floorlist, n_layer):
        self.ceillist = ceillist
        self.floorlist = floorlist
        self.n_layer = n_layer
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(self.n_layer):
            w = module.mask[i].data
            if i in self.ceillist: w.ceil_()
            elif i in self.floorlist: w.floor_()
            else: w.round_()

class TailClipper(object):
    def __init__(self, nclip_list, n_layer):
        self.nclip_list = nclip_list
        self.n_layer = n_layer
    def __call__(self, module):
        if hasattr(module, 'mask'):
          for i in range(self.n_layer):
            w = module.mask[i].data
            w[torch.topk(torch.abs(w),self.nclip_list[i],largest=False)[1]] = 0 # make abs(w) smallest k values to be zero
            w[w != 0] = 1 # make remaining w to be 1

