import time
import torch
import torch.nn as nn
from torch.optim import Adamax
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


try:
    from .syncbatchnorm import SynchronizedBatchNorm1d
except (ImportError, ModuleNotFoundError) as err:
    from syncbatchnorm import SynchronizedBatchNorm1d


BatchNorm1d = SynchronizedBatchNorm1d


def compute_padding(kernel_size):
    if kernel_size % 2 == 0:  # Even number
        raise ValueError("kernel_size should be odd number.")
        
    return (kernel_size - 1) // 2
    


def layer_type1(in_channels, out_channels, kernel_size=3):
    padding = compute_padding(kernel_size)
    layer = nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size=kernel_size, stride=1, padding=padding),
                BatchNorm1d(out_channels),
                nn.ReLU())
    return layer
            
def layer_type2(in_channels, out_channels, kernel_size=3):
    padding = compute_padding(kernel_size)
    layer = nn.Sequential(
                layer_type1(in_channels, out_channels),
                nn.Conv1d(out_channels,
                          out_channels,
                          kernel_size=kernel_size, stride=1, padding=padding),
                BatchNorm1d(out_channels))
    return layer
    
def sublayer1_type3(in_channels, out_channels, kernel_size=3):
    padding = compute_padding(kernel_size)
    layer = nn.Sequential(
                layer_type1(in_channels, out_channels),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=kernel_size, stride=1, padding=padding),
                BatchNorm1d(out_channels),
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1))                 
    return layer

def sublayer2_type3(in_channels, out_channels, kernel_size=3):
    padding = compute_padding(kernel_size)
    layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=kernel_size, stride=2,
                          padding=padding),
                BatchNorm1d(out_channels),)                    
    return layer

def layer_type4(in_channels, out_channels, kernel_size=3):
    padding = compute_padding(kernel_size)
    layer = nn.Sequential(
                layer_type1(in_channels, out_channels),
                nn.Conv1d(out_channels, out_channels,
                          kernel_size=kernel_size, stride=1, padding=padding),
                BatchNorm1d(out_channels),
                nn.AdaptiveAvgPool1d(1),  # Global average pooling (GAP)
                Flatten(1))
    return layer




class SRNet(nn.Module):
   
    @staticmethod
    def get_optimizer(model, lr):        
        return RAdam(model.parameters(), lr=lr, weight_decay=1e-5)
        #return Adamax(model.parameters(), lr=lr, weight_decay=1e-5)

    
    @staticmethod
    def get_lr_scheduler(optimizer):        
        return CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)
    
    def __str__(self):
        return self._name

    def __init__(self, kernel_size=None):
        super().__init__()
        
        if not kernel_size:
            self._kernel_size = 3
            self._name = "srnet"
        else:
            if type(kernel_size) != int:                
                raise ValueError("kernel_size should be integer, "\
                                 "not {} type."%type(kernel_size))
            self._kernel_size = kernel_size
            self._name = "srnet-{}".format(kernel_size)
            
        
        self.layer1 = layer_type1(1, 64, kernel_size)
        self.layer2 = layer_type1(64, 16, kernel_size)
        
        self.layer3 = layer_type2(16, 16, kernel_size)
        self.layer4 = layer_type2(16, 16, kernel_size)
        self.layer5 = layer_type2(16, 16, kernel_size)
        self.layer6 = layer_type2(16, 16, kernel_size)
        self.layer7 = layer_type2(16, 16, kernel_size)
        
        self.layer8_sub1 = sublayer1_type3(16, 16, kernel_size)
        self.layer8_sub2 = sublayer2_type3(16, 16, kernel_size)
        
        self.layer9_sub1 = sublayer1_type3(16, 64, kernel_size)
        self.layer9_sub2 = sublayer2_type3(16, 64, kernel_size)
        
        self.layer10_sub1 = sublayer1_type3(64, 128, kernel_size)
        self.layer10_sub2 = sublayer2_type3(64, 128, kernel_size)
        
        self.layer11_sub1 = sublayer1_type3(128, 256, kernel_size)
        self.layer11_sub2 = sublayer2_type3(128, 256, kernel_size)
        
        self.layer12 = layer_type4(256, 512, kernel_size)
        
        self.layer13 = nn.Linear(512, 2)
              
        self.initialize_parameters()
        
    def initialize_parameters(self):            
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_in',
                        nonlinearity='relu')
                nn.init.constant_(m.bias.data, val=0.2)  
            elif isinstance(m, nn.Linear):
                # Zero mean Gaussian with std 0.01
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out_1 = self.layer1(x) 
        out_2 = self.layer2(out_1)
        
        out_3 = self.layer3(out_2) + out_2
        out_4 = self.layer4(out_3) + out_3
        out_5 = self.layer5(out_4) + out_4
        out_6 = self.layer6(out_5) + out_5
        out_7 = self.layer7(out_6) + out_6
        
        out_8 = self.layer8_sub1(out_7) + self.layer8_sub2(out_7)
        out_9 = self.layer9_sub1(out_8) + self.layer9_sub2(out_8)
        out_10 = self.layer10_sub1(out_9) + self.layer10_sub2(out_9)
        out_11 = self.layer11_sub1(out_10) + self.layer11_sub2(out_10)
        out_12 = self.layer12(out_11)
        out_13 = self.layer13(out_12)      
        
        return out_13

if __name__ == "__main__":   
    import gc
    x = torch.randn(1, 1, 16000)
    for i in range(1, 6):
        ks = 2*i + 1
        model = SRNet(kernel_size=ks)
        t_beg = time.time()    
        out = model(x)
        t_end = time.time()        
        print("SRNet(ks=%d) execution time:"%(ks), t_end - t_beg)
        print("Model output:", out)
        print()

        
    for n in [5, 7, 9, 11]:
        ks = n**2
        model = SRNet(kernel_size=ks)
        t_beg = time.time()    
        out = model(x)
        t_end = time.time()        
        print("SRNet(ks=%d) execution time:"%(ks), t_end - t_beg)
        print("Model output:", out)
        print()
