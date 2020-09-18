# -*- coding: utf-8 -*-
import os
from os.path import join as pjoin
import time

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import time
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau 
try:
    from .layers import HPF
    from .layers import TLU
    from .layers import BitplaneSeparation
    from .layers import FeatureExtractionLayer
    from .layers import ConvLayer
    from .layers import ClassificationLayer
    from .radam import RAdam
except (ImportError, ModuleNotFoundError) as err:
    from layers import HPF
    from layers import TLU
    from layers import BitplaneSeparation
    from layers import FeatureExtractionLayer
    from layers import ConvLayer
    from layers import ClassificationLayer
    from radam import RAdam
    

    
class BSNet(nn.Module):
    
    
    @staticmethod
    def get_optimizer(model, lr):        
        return RAdam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    @staticmethod
    def get_lr_scheduler(optimizer):        
        return CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    
    def __str__(self):
        return self._name
    
    def __init__(self,
                 mode=None,
                 n_bits=16,
                 ch_in=1,
                 ch_init=128,
                 n_convs=4,
                 n_classes=2,                 
                 attention="sqex",
                 conv_type="stdz"):
        super().__init__()
        
        mode = mode.lower()
        if mode == "bsn":
            self.layer0 = BitplaneSeparation(ch_in=ch_in,
                                             n_bits=n_bits,
                                             no_weight=True,
                                             inv_weight=False)        
            self.layer1_fe = FeatureExtractionLayer(n_bits, ch_init, n_convs)
            
        elif mode == "bsn-hpf":
            self.layer0 = HPF(ch_in, 4, 5, padding=2)            
            self.layer1_fe = FeatureExtractionLayer(4, ch_init, n_convs)
            
        elif mode == "bsn-hpf-tlu":
            self.layer0 = nn.Sequential(HPF(ch_in, 4, 5, padding=2),
                                        nn.Conv1d(4, 8, 1),
                                        TLU(3.0))        
            self.layer1_fe = FeatureExtractionLayer(8, ch_init, n_convs)
        elif mode == "bsn-nobs":
            self.layer0 = nn.Conv1d(1, 8, 1)
            self.layer1_fe = FeatureExtractionLayer(8, ch_init, n_convs)
        else:
            raise ValueError("mode should be designated: %s"%(mode))
            
        self._name = mode
        
        self.layer2_conv = ConvLayer(ch_init, 2*ch_init,
                                                conv_type=conv_type)
        self.layer3_conv = ConvLayer(2*ch_init, 2*ch_init,
                                                conv_type=conv_type)
        self.layer4_conv = ConvLayer(2*ch_init, 2*ch_init,
                                                conv_type=conv_type,
                                                attention=attention)
        
        self.layer5_conv = ConvLayer(2*ch_init, 4*ch_init, stride=2,
                                                conv_type=conv_type)
        self.layer6_conv = ConvLayer(4*ch_init, 4*ch_init,
                                                conv_type=conv_type)
        self.layer7_conv = ConvLayer(4*ch_init, 4*ch_init,
                                                conv_type=conv_type,
                                                attention=attention)
        
        self.layer8_conv = ConvLayer(4*ch_init, 8*ch_init, stride=2,
                                                conv_type=conv_type)
        self.layer9_conv = ConvLayer(8*ch_init, 8*ch_init,
                                                conv_type=conv_type)
        self.layer10_conv = ConvLayer(8*ch_init, 8*ch_init,
                                                 conv_type=conv_type,
                                                 attention=attention)
        
        self.layer11_conv = ConvLayer(8*ch_init, 16*ch_init, stride=2,
                                                 conv_type=conv_type)
        self.layer12_conv = ConvLayer(16*ch_init, 16*ch_init,
                                                 conv_type=conv_type)
        self.layer13_conv = ConvLayer(16*ch_init, 16*ch_init,
                                                 conv_type=conv_type,
                                                 attention=attention)
        
        self.layer14_cls = ClassificationLayer(16*ch_init,
                                               n_classes=n_classes)
        
        self.initialize_parameters()

    def forward(self, x):
        
        y0 = self.layer0(x)
        y1 = self.layer1_fe(y0)
        
        y2 = self.layer2_conv(y1)
        y3 = self.layer3_conv(y2)        
        y4 = self.layer4_conv(y3)
        
        y5 = self.layer5_conv(y4)
        y6 = self.layer6_conv(y5)        
        y7 = self.layer7_conv(y6)
        
        y8 = self.layer8_conv(y7)
        y9 = self.layer9_conv(y8)
        y10 = self.layer10_conv(y9)
        
        y11 = self.layer11_conv(y10)
        y12 = self.layer12_conv(y11)
        y13 = self.layer13_conv(y12)
                
        y14 = self.layer14_cls(y13)
            
        return y14
    
    def initialize_parameters(self):            
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, val=1e-4)  
            elif isinstance(m, nn.Linear):                
                nn.init.normal_(m.weight, 0.0, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-4)
                    
if __name__ == "__main__": 
    n_batches = 1
    n_ch = 1
    
    x = torch.randn(n_batches, n_ch, 16000)
    # bps = BitplaneSeparation(ch_in=1)
    # y = bps(x)
    
    model = BSNet(mode="bsn")
    print(model)
    t_beg = time.time()
    y = model(x)
    t_end = time.time()
    print(y)
    print(y.shape)
    print("Time elapsed:", t_end - t_beg)

    """
    model = BSNet(mode="bsn-hpf")
    y = model(x)
    print(y)
    
    model = BSNet(mode="bsn-hpf-tlu")
    y = model(x)
    print(y)
    
    model = BSNet(mode="bsn-nobs")
    y = model(x)
    print(y)
    
    
    # bsn-canoconv
    model = BSNet(mode="bsn", conv_type="cano")
    y = model(x)
    print(y)
    
    # bsn-noatt
    model = BSNet(mode="bsn", attention=None)
    y = model(x)
    print(y)
    
    # bsn-canonconv-noatt
    model = BSNet(mode="bsn", conv_type="cano", attention=None)
    y = model(x)
    print(y)
    
    # bsn-stdzconv-noatt
    model = BSNet(mode="bsn", conv_type="stdz", attention=None)
    y = model(x)
    print(y)
    """
