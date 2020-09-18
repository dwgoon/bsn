import os
from os.path import join as pjoin
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from .radam import RAdam
except (ImportError, ModuleNotFoundError) as err:
    from radam import RAdam

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
                 in_channels,
                 out_channels,
                 kernel_size,
                 *args,
                 **kwargs):        
        super(HPF, self).__init__(in_channels,
                                  out_channels,
                                  kernel_size,
                                  *args, bias=False, **kwargs)

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
        
        # The following settings does not allow training HPF.
        #self.bias.data.fill_(0)
        #self.hpf.bias.requires_grad = False  

class TLU(nn.Module):
    def __init__(self, thr=3.0):
        """truncated linear unit (TLU)
        """
        super(TLU, self).__init__()
        self.thr = thr

    def forward(self, x):        
        return x.clamp(-self.thr, self.thr)  #torch.min(torch.max(x, -self.thr), self.thr)        

class Group1(nn.Module):
    
    def __init__(self):
        super(Group1, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(4, 8, 1),
                                    TLU(3.0),
                                    nn.Conv1d(8, 8, 5, padding=2),
                                    nn.Conv1d(8, 16, 1))
    def forward(self, x):
        return self.module(x)


class Group2(nn.Module):
    
    def __init__(self):
        super(Group2, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(16, 16, 5, padding=2),
                                    nn.ReLU(),
                                    nn.Conv1d(16, 32, 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(3, stride=2, padding=1))
    def forward(self, x):
        return self.module(x)


class Group3(nn.Module):
    
    def __init__(self):
        super(Group3, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2),
                                    nn.ReLU(),
                                    nn.Conv1d(32, 64, 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(3, stride=2, padding=1))
    def forward(self, x):
        return self.module(x)


class Group4(nn.Module):
    
    def __init__(self):
        super(Group4, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(64, 64, 5, padding=2),
                                    nn.ReLU(),
                                    nn.Conv1d(64, 128, 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(3, stride=2, padding=1))
    def forward(self, x):
        return self.module(x)


class Group5(nn.Module):
    
    def __init__(self):
        super(Group5, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(128, 128, 5, padding=2),
                                    nn.ReLU(),
                                    nn.Conv1d(128, 256, 1),
                                    nn.ReLU(),
                                    nn.AvgPool1d(3, stride=2, padding=1))
    def forward(self, x):
        return self.module(x)


class Group6(nn.Module):
    
    def __init__(self):
        super(Group6, self).__init__()
        self.module = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2),
                                    nn.ReLU(),
                                    nn.Conv1d(256, 512, 1),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(1))
    def forward(self, x):
        return self.module(x)
    

class Classifier(nn.Module):
    
    def __init__(self, n_classes=2):
        super(Classifier, self).__init__()
        self.module = nn.Sequential(Flatten(1),
                                    nn.Linear(512, n_classes))
        
    def forward(self, x):
        return self.module(x)
    

class LinNet(nn.Module):

    @staticmethod
    def get_optimizer(model, lr):        
        return RAdam(model.parameters(), lr=lr, weight_decay=1e-5)

    
    @staticmethod
    def get_lr_scheduler(optimizer):        
        return CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)

    def __str__(self):
        return self._name

    def __init__(self, n_classes=2):
        super(LinNet, self).__init__()
        
        self._name = "linnet"

        # HPF
        self.hpf = HPF(1, 4, 5, padding=2)
        self.group1 = Group1()
        self.group2 = Group2()
        self.group3 = Group3()
        self.group4 = Group4()
        self.group5 = Group5()
        self.group6 = Group6()
        self.classifier = Classifier(n_classes)
        
        self.initialize_parameters()
    
    def forward(self, x):
        y = self.hpf(x)    
        g1 = self.group1(y)
        g2 = self.group2(g1)
        g3 = self.group3(g2)
        g4 = self.group4(g3)
        g5 = self.group5(g4)
        g6 = self.group6(g5)
        logits = self.classifier(g6)
        
        return logits        
       
    def initialize_parameters(self):
        """
        In the original paper, Lin et al.
        
        Conv1d: Xavier uniform initializer with zero biases
        
        """          
        """
        [Original]
        for m in self.modules():
            if isinstance(m, HPF):
                self.hpf.initialize_parameters()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias.data, val=1e-3)  
            elif isinstance(m, nn.Linear):
                # Zero mean Gaussian with std 0.01
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-3)
        """
        
        # Following settings is the same with that of BSN.
        for m in self.modules():
            if isinstance(m, HPF):
                self.hpf.initialize_parameters()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)  
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
                    
    def initialize_curriculum_learning(self):
        for m in self.modules():
            if isinstance(m, HPF):
                self.hpf.initialize_parameters()

            elif isinstance(m, nn.Linear):
                # Zero mean Gaussian with std 0.01
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-3)
        
                    
if __name__ == "__main__":   
    model = LinNet()    
    n_ch = 1
    for i in range(1, 2):
        x = torch.randn(1, n_ch, i*16000)        
        t_beg = time.time()    
        out = model(x)
        t_end = time.time()
        print("LinNet model output:", out)
        print("Execution time:", t_end - t_beg)        
    # end of for
    
