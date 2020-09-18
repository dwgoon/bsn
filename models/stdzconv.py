import torch
import torch.nn as nn
import torch.nn.functional as F

        
class StdzConv1d(nn.Conv1d):
    """Standardized convolution

       https://arxiv.org/abs/1903.10520
       https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        weight = weight - weight_mean
        #std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# end of def class
