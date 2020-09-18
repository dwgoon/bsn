import torch.nn as nn

from .swish import Swish
from .conv import Conv1d
    
class SqEx1d(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                      nn.Linear(ch, ch // reduction, bias=True),
                      nn.ReLU(inplace=True),
                      nn.Linear(ch // reduction, ch, bias=True),
                      nn.Sigmoid())

    def forward(self, x):
        b, c, _  = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
    
class ConvSqEx1d(nn.Module):

    def __init__(self, ch, reduction=16, activation='relu'):
        super(ConvSqEx1d, self).__init__()
        
        activation = activation.lower()
        
        if activation == 'swish':
            Activation = Swish
        elif activation == 'relu':
            Activation = nn.ReLU
        else:
            err_msg = "{} is now valid for activation.".format(activation)
            raise ValueError(err_msg)
                
        ch_reduced = ch // reduction
        self.se = nn.Sequential(
                      nn.AdaptiveAvgPool1d(1),
                      Conv1d(ch, ch_reduced, 1),
                      Activation(),
                      Conv1d(ch_reduced, ch, 1),
                      nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)
