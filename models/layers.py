from os.path import join as pjoin

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .conv import Conv1d
from .stdzconv import StdzConv1d
from .sqex import SqEx1d, ConvSqEx1d

try:
    from torch.nn import Flatten
except ImportError:
    class Flatten(nn.Module):
        __constants__ = ['start_dim', 'end_dim']

        def __init__(self, start_dim=1, end_dim=-1):
            super(Flatten, self).__init__()
            self.start_dim = start_dim
            self.end_dim = end_dim

        def forward(self, input):
            return input.flatten(self.start_dim, self.end_dim)


class HPF(nn.Conv1d):      
    
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size=5,
                 *args,
                 **kwargs):
        
        super().__init__(ch_in,
                         ch_out,
                         kernel_size,
                         *args,
                         bias=False, 
                         **kwargs)

        self.hpf_kernel = np.array([[[ 1, -1,  0,  0,  0]],
                                    [[ 1, -2,  1,  0,  0]],
                                    [[ 1, -3,  3, -1,  0]],
                                    [[ 1, -4,  6, -4,  1]]])
        self.weight.data = torch.tensor(self.hpf_kernel,
                                        dtype=self.weight.dtype)
        

    def initialize_parameters(self):
        device = next(iter(self.parameters())).device
        self.weight.data = torch.tensor(self.hpf_kernel,
                                        dtype=self.weight.dtype,
                                        device=device)
        
        
class TLU(nn.Module):
    def __init__(self, thr=3.0):
        """truncated linear unit (TLU)
        """
        super().__init__()
        self.thr = thr

    def forward(self, x):        
        return x.clamp(-self.thr, self.thr)
    
    
class BitplaneSeparation(nn.Module):
    
    def __init__(self, ch_in, n_bits=16, no_weight=True, inv_weight=False, eps=1e-3):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_in * n_bits
        self.n_bits = n_bits
        self.no_weight = no_weight
        self.inv_weight = inv_weight  # Inverse weight        
        self.eps = eps
        
    def forward(self, x):
        n_batch, n_ch, _ = x.shape
        y = x.to(torch.int32)  # Conversion data type
        bitplanes = []
        for c in range(n_ch):
            y_ch = y[:, c, :]
            for i in range(self.n_bits):  # Number of bit
                if self.no_weight:                    
                    plane = (y_ch & 1).to(x.dtype) + self.eps
                else:
                    w = (i+1) if not self.inv_weight else (self.n_bits-i)
                    plane = w*(y_ch & 1).to(x.dtype) + self.eps
                
                bitplanes.append(plane)
                y_ch = y_ch >> 1

        return torch.stack(bitplanes, dim=1)


class InitActivation(nn.Module):
    def __init__(self):
        super(InitActivation, self).__init__()
        self.module = nn.ReLU()

    def forward(self, x):
        return self.module(x)
    
    
class IntermediateActivation(nn.Module):
    def __init__(self):
        super(IntermediateActivation, self).__init__()
        self.module = nn.ReLU()

    def forward(self, x):
        return self.module(x)    

        
class FeatureExtractionLayer(nn.Module):    
    def __init__(self, ch_in=16, ch_out=128, n_convs=8, attention='sqex'):
        super().__init__()
        
        self.n_convs = n_convs
        
        if ch_out % n_convs != 0:
            raise ValueError("ch_out should be divided by n_convs.")
            
        ch_out_each = ch_out // n_convs
        
        self.convs = nn.ModuleList()
        
        for i in range(n_convs):
            self.convs.append(Conv1d(ch_in, ch_out_each,
                                     kernel_size=2*i+1,
                                     padding=i))
        
        self.norm = nn.GroupNorm(num_groups=n_convs,
                                 num_channels=ch_out)
        
        
        attention = attention.lower()
        if attention == 'sqex':
            self.attention = SqEx1d(ch_out)
        else:
            self.attention = None

    def forward(self, x):
        ys = []
        for i in range(self.n_convs):
            conv = self.convs[i]
            ys.append(conv(x))
        
        out = self.norm(torch.cat(ys, dim=1))
        if self.attention:
            out = self.attention(out)

        return out


class ConvLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride=1,
                 groups=4,
                 conv_type="stdz",
                 attention=None):
        super().__init__()        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.groups = groups
        self.stride = stride

        self.attention = attention
        modules = [Conv1d(ch_in, ch_out,
                          kernel_size=3,
                          padding=1,
                          stride=stride,
                          groups=groups,
                          conv_type=conv_type),
                   nn.GroupNorm(num_groups=groups,
                                num_channels=ch_out),
                   nn.ReLU(),
                   Conv1d(ch_out, ch_out,
                          kernel_size=1,
                          padding=0,
                          groups=groups,
                          conv_type=conv_type),
                   nn.ReLU(),
                   nn.GroupNorm(num_groups=groups,
                                num_channels=ch_out)]
        
        self.attention = attention.lower() if attention else None
        if self.attention == 'cbam': 
            modules.append(CBAM1d(ch_out))
        elif self.attention == 'sqex':
            modules.append(SqEx1d(ch_out))
        
        self.module = nn.Sequential(*modules)
        
    def forward(self, x):
        y = self.module(x)
        if self.ch_in == self.ch_out and x.shape[-1] == y.shape[-1]:
            return  y + x
        
        return y


class ClassificationLayer(nn.Module):
    
    def __init__(self, ch_in, n_classes=2):
        super().__init__()
        self.module = nn.Sequential(nn.AdaptiveAvgPool1d(1), # Global average pooling (GAP)
                                    Flatten(1),                    
                                    nn.Linear(ch_in, 2))
        
    def forward(self, x):
        return self.module(x)
    

class LightWeightConvLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride=1,
                 groups=4,
                 conv_type="stdz",
                 attention=None):
        super().__init__()        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.groups = groups
        self.stride = stride

        self.attention = attention
        modules = [Conv1d(ch_in, ch_out,
                          kernel_size=3,
                          padding=1,
                          stride=stride,
                          groups=groups,
                          conv_type=conv_type),
                   nn.GroupNorm(num_groups=groups,
                                num_channels=ch_out),
                   IntermediateActivation(),
                   Conv1d(ch_out, ch_out,
                          kernel_size=1,
                          padding=0,
                          groups=groups,
                          conv_type=conv_type),
                   IntermediateActivation(),
                   nn.GroupNorm(num_groups=groups,
                                num_channels=ch_out)]
        
        self.attention = attention.lower() if attention else None
        if self.attention == 'cbam': 
            modules.append(CBAM1d(ch_out))
        elif self.attention == 'sqex':
            modules.append(SqEx1d(ch_out))
        
        self.module = nn.Sequential(*modules)
        
    def forward(self, x):
        y = self.module(x)
        if self.ch_in == self.ch_out and x.shape[-1] == y.shape[-1]:
            return  y + x
        
        return y
