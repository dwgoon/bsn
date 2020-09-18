import os
import re
from glob import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn



def get_str_now(fstr="%Y%m%d-%H%M%S"):
    dt = datetime.now()
    return dt.strftime(fstr)

def visdom_clear_environments(vis):
    for env_name in vis.get_env_list():
        vis.delete_env(env_name) 

def visdom_create_lineplot_window(vis,
                                  xlabel,
                                  ylabel,
                                  title,
                                  init_y=0.5,
                                  legend=None):
    opts = {'xlabel':xlabel,
            'ylabel':ylabel,
            'title':title,}
    
    if legend:
        opts['legend'] = legend
    
    return vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.tensor([init_y]).cpu(),
                    opts=opts)



def create_visdom_histogram_window(vis,
                                   xlabel,
                                   ylabel,
                                   title,
                                   init_y=0.5,
                                   legend=None):
    opts = {'xlabel':xlabel,
            'ylabel':ylabel,
            'title':title,}
    
    if legend:
        opts['legend'] = legend
    
    return vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.tensor([init_y]).cpu(),
                    opts=opts)

def get_best_ckpt_with_criterion(dpath_ckpt, policy='max_epoch'): #max_epoch=True, min_loss=False):
    
    criterion_func = None
    policy = policy.lower()
    if policy not in ('max_epoch', 'min_loss', 'max_acc'):
        err_msg = "One of (max_epoch|min_loss|max_acc) should be selected."
        raise ValueError(err_msg)
        
    elif policy == 'max_epoch':
        ind_group = 1
        criterion_func = lambda new, old: new > old  # Max epoch
        type_conv_func = lambda x: int(x)
        init_best = lambda : 0    
    elif policy == 'min_loss':
        ind_group = 2
        criterion_func = lambda new, old: new < old  # Min loss
        type_conv_func = lambda x: float(x)
        init_best = lambda : np.finfo(np.float32).max
    elif policy == 'max_acc':
        ind_group = 3
        criterion_func = lambda new, old: new > old  # Max accuracy
        type_conv_func = lambda x: float(x)
        init_best = lambda : 0.0
    else:
        raise RuntimeError("This point cannot be reached...")
    
    curr_best = init_best()
    pat = re.compile("ckpt_.+_epoch-(\d+)_loss-(\d+.\d+)_acc-(\d+.\d+)\.pth.tar")
    fname = None
    for entity in os.listdir(dpath_ckpt):
        if not entity.endswith(".pth.tar"):
            continue
        
        mat = pat.search(entity)
        if mat:            
            val = type_conv_func(mat.group(ind_group))            
            if criterion_func(val, curr_best):
                curr_best = val
                fname = entity
    # end of for
    if not fname:
        return None
    
    return os.path.join(dpath_ckpt, fname)
    

def get_ckpt(dpath_ckpt, policy='min-loss'): #max_epoch=True, min_loss=False):
    
    #criterion_func = None
    policy = policy.lower()
    if policy not in ('max-epoch', 'min-loss', 'max-acc'):
        err_msg = "One of (max-epoch|min-loss|max-acc) should be selected."
        raise ValueError(err_msg)
        
    pat = "ckpt_{}".format(policy)
    fname = None
    for entity in os.listdir(dpath_ckpt):
        if entity.endswith(".pth.tar") and pat in entity:
            fname = entity
            break
    # end of for
    if not fname:
        return None
    
    return os.path.join(dpath_ckpt, fname)
    

def load_model(fpath, model, optimizer=None):
    ckpt = torch.load(fpath)
    
    if isinstance(model, nn.DataParallel):
        model = model.module
    else:
        model = model

    model.load_state_dict(ckpt['model'])

    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
        return  ckpt, model, optimizer

    return ckpt, model

def save_model(model,
               fpath=None,
               dataset_name=None,
               model_name=None,
               epoch=None,
               loss=None,
               acc=None,               
               optimizer=None,
               fname=None,
               dpath="checkpoints"):
    
    if not os.path.exists(dpath):
        os.mkdir(dpath)
        
    if isinstance(model, nn.DataParallel):
        model = model.module
    else:
        model = model

    ckpt = {}
    ckpt['model'] = model.state_dict()
    if optimizer:
        ckpt['optimizer'] = optimizer

    if fpath:
        torch.save(ckpt, fpath)
        return        

    if not fname:        
        fstr_fname = "ckpt_{}_{}_epoch-{}_loss-{:.04}_acc-{:.04}.pth.tar"
        fname = fstr_fname.format(dataset_name,
                                  model_name,
                                  epoch,
                                  loss,
                                  acc)
            
    
    fpath = os.path.join(dpath, fname)
    torch.save(ckpt, fpath)
    return fpath
       

def accuracy_thr(output, target):
    pred = (output > 0.5).to(torch.float32)
    acc = 1 - torch.mean(torch.abs(pred - target))
    return acc.item()


def accuracy_top1(logits, target):
    ind = logits.topk(1).indices.view(-1)
    return ind.eq(target).float().mean().item()

def accuracy_topk(logits, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append( (correct_k.mul_(1.0 / batch_size)).item() )
    return res
