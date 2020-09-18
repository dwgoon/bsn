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
    from .layers import BitplaneSeparation
    from .layers import FeatureExtractionLayer
    from .layers import ConvLayer
    from .layers import ClassificationLayer
    from .radam import RAdam
except (ImportError, ModuleNotFoundError) as err:
    from layers import HPF
    from layers import BitplaneSeparation
    from layers import FeatureExtractionLayer
    from layers import ConvLayer
    from layers import ClassificationLayer
    from radam import RAdam


class HpfConvNet(nn.Module):
    
    
    @staticmethod
    def get_optimizer(model, lr):        
        return RAdam(model.parameters(), lr=lr, weight_decay=1e-5)
        #return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    
    @staticmethod
    def get_lr_scheduler(optimizer):        
        return CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
        #return None
#        return ReduceLROnPlateau(optimizer,
#                                 factor=0.5,
#                                 patience=10,
#                                 threshold=1e-4,
#                                 min_lr=1e-7,
#                                 verbose=True)
    
    
    def __init__(self,
                 ch_in=1,
                 ch_init=128,
                 n_convs=4,
                 n_classes=2,
                 attention=None):
        super(HpfConvNet, self).__init__()
        
        self.layer0_preproc = HPF(ch_in=ch_in,
                                  ch_out=4,
                                  kernel_size=5,
                                  padding=2)        
        
        self.layer1_fe_type1 = FeatureExtractionLayer(4, ch_init, n_convs)
        
        self.layer2_conv_type1 = ConvLayer(ch_init, 2*ch_init)
        self.layer3_conv_type1 = ConvLayer(2*ch_init, 2*ch_init)
        self.layer4_conv_type1 = ConvLayer(2*ch_init, 2*ch_init, attention='sqex')
        
        self.layer5_conv_type1 = ConvLayer(2*ch_init, 4*ch_init, stride=2)
        self.layer6_conv_type1 = ConvLayer(4*ch_init, 4*ch_init)
        self.layer7_conv_type1 = ConvLayer(4*ch_init, 4*ch_init, attention='sqex')
        
        self.layer8_conv_type1 = ConvLayer(4*ch_init, 8*ch_init, stride=2)
        self.layer9_conv_type1 = ConvLayer(8*ch_init, 8*ch_init)
        self.layer10_conv_type1 = ConvLayer(8*ch_init, 8*ch_init, attention='sqex')
        
        self.layer11_conv_type1 = ConvLayer(8*ch_init, 16*ch_init, stride=2)
        self.layer12_conv_type1 = ConvLayer(16*ch_init, 16*ch_init)
        self.layer13_conv_type1 = ConvLayer(16*ch_init, 16*ch_init, attention='sqex')
        
        self.layer14_cls_type1 = ClassificationLayer(16*ch_init,
                                                          n_classes=n_classes)
        
        self.initialize_parameters()

    def forward(self, x):
                
        y0 = self.layer0_preproc(x)
        y1 = self.layer1_fe_type1(y0)        
        
        y2 = self.layer2_conv_type1(y1)
        y3 = self.layer3_conv_type1(y2)        
        y4 = self.layer4_conv_type1(y3)
        
        y5 = self.layer5_conv_type1(y4)
        y6 = self.layer6_conv_type1(y5)        
        y7 = self.layer7_conv_type1(y6)
        
        y8 = self.layer8_conv_type1(y7)
        y9 = self.layer9_conv_type1(y8)
        y10 = self.layer10_conv_type1(y9)
        
        y11 = self.layer11_conv_type1(y10)
        y12 = self.layer12_conv_type1(y11)
        y13 = self.layer13_conv_type1(y12)
                
        y14 = self.layer14_cls_type1(y13)
            
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
                # Zero mean Gaussian with std 0.01
                nn.init.normal_(m.weight, 0.0, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-4)
                    
if __name__ == "__main__": 
    n_batches = 1
    n_ch = 1
    
    x = torch.randn(n_batches, n_ch, 16000)
    bps = BitplaneSeparation(ch_in=1)
    y = bps(x)
    
    model = HpfConvNet(attention='sqex')
    print(model)
    t_beg = time.time()
    y = model(x)
    t_end = time.time()
    print(y)
    print(y.shape)
    print("Time elapsed:", t_end - t_beg)
    
    
    #import torchsummary
    #torchsummary.summary(model, x.shape[1:], device='cpu')
