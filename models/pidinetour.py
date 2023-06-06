import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Conv2d
from .config import config_model, config_model_converted




import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Conv2d
from .config import config_model, config_model_converted
import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com> common






###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()


        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, dilation=1),
            
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 64, 1,padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

        self.init_param()

    def forward(self, x1, x2, x3):
        
  
        H, W = x1.size()[2:]

        outputs=[x3]

        #print('x',x1.shape,x2.shape,x3.shape,x.shape)
        y1 = self.stage1(x3)
        y2 = self.stage2(x3)
        y3 = self.stage3(x3)
        #y4 = self.stage4(x3)
        #print('y4 ',y1.shape,y2.shape,y3.shape)

        y = torch.cat((y1,y2,y3), dim=1)
        #print(y.shape)
        y = self.res(y)
        outputs.append(y)
        #print(y.shape)
        #print(len(outputs))
        return outputs

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def conv_layers(inp, oup, dilation):
    #if dilation:
    d_rate = dilation
    #else:
    #    d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


def feature_transform(inp, oup):
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    return nn.MaxPool2d(kernel_size=3, stride=2)

class block_V(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(block_V, self).__init__()
        

        self.conv0_0 = conv_layers(in_channels, in_channels,2)


        self.pool0 = pool_layers()
        self.conv1_0 = conv_layers(in_channels, in_channels,2)
        self.conv1_1 = conv_layers(in_channels, in_channels,2)

        self.pool1 = pool_layers()
        self.conv2_0 = conv_layers(in_channels, in_channels,2)
        self.conv2_1 = conv_layers(in_channels, in_channels,2)
        
        self.pool2 = pool_layers()
        self.conv3_0 = conv_layers(in_channels, in_channels,2)
        self.conv3_1 = conv_layers(in_channels, in_channels,2)

        
        
    def forward(self, x):
        H, W = x.size()[2:]

        x = self.conv0_0(x)

        x = self.pool0(x)
        x = self.conv1_0(x)      
        x1 = self.conv1_1(x)
       
        x = self.pool1(x1)      
        x = self.conv2_0(x)
        x2 = self.conv2_1(x)
             
        x = self.pool2(x2)
        x = self.conv3_0(x)       
        x3 = self.conv3_1(x)
        

        return [x1,x2,x3]

class block_O(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(block_O, self).__init__()

        self.conv1_0 = conv_layers(in_channels, in_channels,2)
        self.conv1_1 = conv_layers(in_channels, in_channels,2)
   
        self.conv2_0 = conv_layers(in_channels, in_channels,2)
        self.conv2_1 = conv_layers(in_channels, in_channels,2)
              
        self.conv3_0 = conv_layers(in_channels, in_channels,2)
        self.conv3_1 = conv_layers(in_channels, in_channels,2)
        
    def forward(self, x):

        x = self.conv1_0(x)      
        x1 = self.conv1_1(x)
             
        x = self.conv2_0(x1)
        x2 = self.conv2_1(x)
             
        x = self.conv3_0(x2)       
        x3 = self.conv3_1(x)
        
        return [x1,x2,x3]
        
class blockbn(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(blockbn, self).__init__()
        #self.norm_layer = get_norm_layer(norm_type='batch')
        norm_layer = get_norm_layer(norm_type='batch')
        

        self.conv0_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=2,padding=2))
       

        self.pool0 = pool_layers()
        

        self.conv1_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=2,padding=2))
        
        self.pool1 = pool_layers()

        self.conv2_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=2,padding=2))
  
        #self.conv2_2 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool2 = pool_layers()


        self.classifier = nn.Conv2d(in_channels*3, 3, kernel_size=1)
    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=1, kernel_size=3,stride=1, padding=2,bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, dilation=padding, bias=False),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)    
        H, W = x.size()[2:]

        x = self.conv0_0(x)
 
       
        x1 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool0(x)

        x = self.conv1_0(x)
        

        
        x2 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool1(x)
      
        x = self.conv2_0(x)
        

        
        x3 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)

        return torch.cat((x1,x2,x3), dim=1)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PiDiNet(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(PiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        self.inplane = inplane

      

        norm_layer = nn.BatchNorm2d

        self.bV=block_V(64)
        self.bO=block_O(64)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)


        
        self.conv1_down = nn.Conv2d(64, 3, 1, padding=1)
        self.conv2_down = nn.Conv2d(64, 3, 1, padding=1)
        self.conv3_down = nn.Conv2d(64, 3, 1, padding=1)
        self.conv4_down = nn.Conv2d(192, 3, 1, padding=1)
        self.conv5_down = nn.Conv2d(192, 3, 1, padding=1)
        self.conv6_down = nn.Conv2d(192, 3, 1, padding=1)
        self.p = nn.MaxPool2d(2, stride=2)


        self.relu = nn.ReLU()
        
        self.score_dsn1 = nn.Sequential(*self._conv_block(21, 1, norm_layer, num_block=1))
        self.score_dsn2 = nn.Sequential(*self._conv_block(21, 1, norm_layer, num_block=1))
        self.score_dsn3 = nn.Sequential(*self._conv_block(21, 1, norm_layer, num_block=1))
        self.score_dsn4 = nn.Sequential(*self._conv_block(21, 1, norm_layer, num_block=1))

        

        self.bn=blockbn(64)
        self.bn1=blockbn(64)
        self.bn2=blockbn(64)
        self.score_final = nn.Conv2d(18, 1, 1)
  
        

        
        nn.init.constant_(self.score_final.weight, 0.25)
        nn.init.constant_(self.score_final.bias, 0)

        print('initialization done')
        
    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=1, kernel_size=1,stride=1,bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, bias=False),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv
    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H1, W1 = x.size()[2:]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_1=self.p(conv1_1)
        H, W = conv1_1.size()[2:]
        #print(conv1_1.shape,x.shape)
        [x1,x2,x3] = self.bO(conv1_1)
        #print(x1.shape,x2.shape,x3.shape)
        [y1,y2,y3] = self.bV(conv1_1)
        #print(y1.shape,y2.shape,y3.shape)


        y1 = F.interpolate(y1, (H, W), mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, (H, W), mode="bilinear", align_corners=False)
        y3 = F.interpolate(y3, (H, W), mode="bilinear", align_corners=False)
        z1 = self.bn(y1+x1)
        z2 = self.bn1(y2+x2)
        z3 = self.bn2(y3+x3)        
        conv1_down = self.conv1_down(x1+y1)#torch.cat((x1,y1), dim=1))
        conv2_down = self.conv2_down(x2+y2)#torch.cat((x2,y2), dim=1))
        conv3_down = self.conv3_down(x3+y3)#torch.cat((x3,y3), dim=1))
        conv4_down = self.conv4_down(z1)#torch.cat((x3,y3), dim=1))
        conv5_down = self.conv5_down(z2)#torch.cat((x3,y3), dim=1))
        conv6_down = self.conv6_down(z3)#torch.cat((x3,y3), dim=1))


        so1_out = conv1_down
        so2_out = conv2_down
        so3_out = conv3_down
        so4_out = conv4_down
        so5_out = conv5_down
        so6_out = conv6_down


        so1 = F.interpolate(so1_out, (H1, W1), mode="bilinear", align_corners=False)
        so2 = F.interpolate(so2_out, (H1, W1), mode="bilinear", align_corners=False)
        so3 = F.interpolate(so3_out, (H1, W1), mode="bilinear", align_corners=False)
        so4 = F.interpolate(so4_out, (H1, W1), mode="bilinear", align_corners=False)
        so5 = F.interpolate(so5_out, (H1, W1), mode="bilinear", align_corners=False)
        so6 = F.interpolate(so6_out, (H1, W1), mode="bilinear", align_corners=False)
        #x1 = F.interpolate(so4, (224, 224), mode="bilinear", align_corners=False)
        #so5=self.swin_unet(x1)
        #so5 =#F.interpolate(conv4_3_down, (H, W), mode="bilinear", align_corners=False)
        fusecat = torch.cat((so1, so2, so3, so4, so5, so6), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, so6, fuse]
        results = [torch.sigmoid(r) for r in results]
        
        
        
        return results[6]





def pidinet_tiny(args):
    pdcs = config_model(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa)

def pidinet_small(args):
    pdcs = config_model(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa)

def pidinet(args):
    pdcs = config_model(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa)



## convert pidinet to vanilla cnn

def pidinet_tiny_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return PiDiNet(20, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_small_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return PiDiNet(30, pdcs, dil=dil, sa=args.sa, convert=True)

def pidinet_converted(args):
    pdcs = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return PiDiNet(60, pdcs, dil=dil, sa=args.sa, convert=True)
