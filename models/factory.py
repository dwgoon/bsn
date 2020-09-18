# -*- coding: utf-8 -*-

from .linnet import LinNet
from .srnet import SRNet
from .bsn import BSNet
from .hcn import HpfConvNet

class ModelFactory:
    
    @staticmethod
    def create(config):
        model_id = config['MODEL'].lower()
        
        if model_id == 'linnet':
            model = LinNet()
        elif model_id.startswith('srnet'): 
            model = SRNet(kernel_size=config["KERNEL_SIZE"])
        elif model_id == 'bsn':
            model = BSNet(n_bits=16, mode="bsn")
        elif model_id == 'bsn-hpf':
            model = BSNet(mode="bsn-hpf")
        elif model_id == 'bsn-hpf-tlu':
            model = BSNet(mode="bsn-hpf-tlu")
        elif model_id == 'bsn-nobs':
            model = BSNet(mode="bsn-nobs")
        elif model_id == 'bsn-noatt':
            model = BSNet(mode="bsn",
                         attention=None,
                         conv_type="stdz")
        elif model_id == 'bsn-canoconv':
            model = BSNet(mode="bsn", conv_type="cano")
        elif model_id == 'bsn-canoconv-noatt':
            model = BSNet(mode="bsn",
                         attention=None,
                         conv_type="cano")

        return model
            
