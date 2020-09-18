import torch
import torch.nn as nn

from .stdzconv import StdzConv1d

class Conv1d(nn.Module):
    def __init__(self, *args, conv_type="stdz", **kwargs):
        super(Conv1d, self).__init__()
        
        if conv_type not in ["cano", "stdz"]:
            raise ValueError("Unknow type: %s"%(conv_type))
        
        if conv_type == "stdz":
            self.module = StdzConv1d(*args, **kwargs)
        elif conv_type == "cano":
            self.module = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        return self.module(x)
        
