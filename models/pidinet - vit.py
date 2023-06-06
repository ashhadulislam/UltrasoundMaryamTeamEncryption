"""
Author: Zhuo Su, Wenzhe Liu
Date: Feb 18, 2021
"""

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



        
class blockbn(nn.Module):
    """
    Compact Dilation Convolution based Module
    """
    def __init__(self, in_channels):
        super(blockbn, self).__init__()
        #self.norm_layer = get_norm_layer(norm_type='batch')
        norm_layer = get_norm_layer(norm_type='batch')
        

        self.conv0_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv0_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool0 = pool_layers()
        

        self.conv1_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv1_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool1 = pool_layers()

        self.conv2_0 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv2_1 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))
        self.conv2_2 = nn.Sequential(*self._conv_block(in_channels, in_channels, norm_layer, num_block=1))

        self.pool2 = pool_layers()


        self.classifier = nn.Conv2d(in_channels*3, 3, kernel_size=1)
    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=1, kernel_size=3, 
        stride=1, padding=1, bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [nn.Conv2d(cur_in_nc, out_nc, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias),
                     norm_layer(out_nc),
                     nn.ReLU(True)]
        return conv
        
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)    
        H, W = x.size()[2:]

        x = self.conv0_0(x)
        x = self.conv0_1(x)
       

        x = self.pool0(x)

        x = self.conv1_0(x)
        
        x = self.conv1_1(x)
        e1 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        x = self.pool1(x)
      
        x = self.conv2_0(x)
        
        x = self.conv2_1(x)
        
        x = self.conv2_2(x)
        e2 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        
        #x = self.pool2(x)

        #x = self.conv3_0(x)
        
        #x = self.conv3_1(x)
        
        #x = self.conv3_2(x)
        #e3 = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        return e1+e2#+e3
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange
import copy

def make_model(parent=False):
    return ipt()

class ipt(nn.Module):
    def __init__(self, ):
        super(ipt, self).__init__()
        
        self.scale_idx = 0
        conv=default_conv
        

        
        n_feats = 64
        kernel_size = 3 
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(3, n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act)
            ) for _ in [2,4]
        ])

        self.body = VisionTransformer(img_dim=256, patch_dim=4, num_channels=n_feats, embedding_dim=n_feats*2*2, num_heads=4, num_layers=4, hidden_dim=n_feats*2*2*4, num_queries = 1, dropout_rate=0, mlp=False,pos_every=False,no_pos=True,no_norm=False)
        self.cov1 =nn.Conv2d(16, 1, kernel_size=1)
        self.cov2 =nn.Conv2d(3, 1, kernel_size=1)
        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, 3, kernel_size)
            ) for s in [2,4]
        ])
        

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        #x = self.sub_mean(x)
        #print(x.shape)
        #x = self.head[self.scale_idx](x)
        #print(x.shape)
        self.scale_idx=1

        res = self.body(x,self.scale_idx)
        #print(res.shape)
        res=self.cov1(res)
        #print(res)
        #res += x
        #res=self.cov2(res) 
        #print(x.shape)

        #x = self.tail[self.scale_idx](res)
        #print(x.shape)
        #x = self.add_mean(res)
        #print(x)

        return res 

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0.1,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos
        
        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear( 48,4096),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(4096, 256),
                nn.Dropout(dropout_rate)
            )
        
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
               
        if self.mlp==False:
            #print(self.linear_encoding(x).shape)
            x = self.dropout_layer1(x)#self.linear_encoding(x))#+self.linear_encoding(x)
            

            query_embed = self.query_embed.weight[query_idx-1].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)

        if self.pos_every:
            x = self.mlp_head(x)
            x = self.encoder(x, pos=pos)
            #x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.mlp_head(x)
            x = self.encoder(x)
            #x = self.decoder(x, x, query_pos=query_embed)

        
        
        if self.mlp==True:
            #print(x.shape)
            x = self.mlp_head(x) #+ x
            #print(x.shape)
        
        #x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x
        x = torch.permute(x, (1, 0, 2))
        #print(x.shape)
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        #print('encoder', src.shape)
        #print(src)
        return src

    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        #print('encoder', tgt.shape)
        #print(src)        
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
        #self.v =  ViT(dim=1024,image_size=256,patch_size=16, num_classes=1, channels=3, depth = 6, heads = 16, mlp_dim = 512)
        self.v=ipt()

        self.f1 = nn.Conv2d(3, 64, kernel_size=3)
        
        self.f2 = nn.Conv2d(64, 3, kernel_size=3,padding=1) 
        self.f4 = nn.Conv2d(64, 1, kernel_size=3,padding=1)
        self.p1_bn = blockbn(64)
        self.p2_bn = blockbn(3)
        #self.sam=SAM(64)

        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.classifier = nn.Conv2d(3, 1, kernel_size=1) # has bias
        
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

        print('initialization done')

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
        H, W = x.size()[2:]
        #e5= self.v(x)
        x = self.f1(x)
        x =self.p1_bn(x)
        x = self.f2(x)
        #print(x.shape)
        #print(e5.shape)
       
        
        e1 = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)
        #e5 = F.interpolate(e5, (H, W), mode="bilinear", align_corners=False)
        #print(e1.shape)
        #print(e5.shape)

        
        
        
        e6= self.v(e1) 
        e7 =self.p2_bn(e6) 
        e7 = F.interpolate(e7, (H, W), mode="bilinear", align_corners=False)        

        #e5 = self.classifier(e5)
        #mean, std, var = torch.mean(e5), torch.std(e5), torch.var(e5)

  

        #e5  = (e5-mean)/std
        #print(t)
        
       
        e1= self.classifier(e1)
        e7= self.classifier(e7)
        e2= e6#self.f3(y)
        e3= e7#self.f3(e5)
          
        #print(e1.shape)
        #print(e2.shape)
        #print(e3.shape)

        outputs = [e1, e2,e3]

        output = self.classifier(torch.cat(outputs, dim=1))
        #if not self.training:
        #    return torch.sigmoid(output)

        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs





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
