import os
from os.path import join as pjoin
import platform
import random
import math
import re
import argparse
import pickle

import yaml
import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

from models import ModelFactory
import utils
from utils import load_model, save_model
from utils import accuracy_top1
from utils import get_ckpt
from dataset import TimitTrainSet, TimitTestSet


parser = argparse.ArgumentParser(description='Parse configruations for training LeeNet.')
parser.add_argument('--config',
                    dest='config',
                    action='store',
                    type=str,
                    help='Designate the file path of configuration file in YAML')

args = parser.parse_args()
fpath_config = args.config

with open(fpath_config, "rt") as fin:
    config = yaml.safe_load(fin)
    hostname = platform.node()
    if 'forensic' in hostname:
        config['SYSTEM_SPECIFIC'] = config['SYSTEM_SPECIFIC']['FORENSIC']
    else:
        config['SYSTEM_SPECIFIC'] = config['SYSTEM_SPECIFIC'][hostname.upper()]

# Configuration
MODEL = config['MODEL']
STEGANOGRAPHY = config['STEGANOGRAPHY']
SEED = config['SEED']
N_REPEATS = config['TEST']['N_REPEATS']
USE_NORM = config['TEST']['USE_NORM']
LOAD_POLICY = config['TEST']['LOAD_POLICY']

BATCH_SIZE = config['SYSTEM_SPECIFIC']['BATCH_SIZE']
N_WORKERS = config['SYSTEM_SPECIFIC']['N_WORKERS']

dpath_cover = config['SYSTEM_SPECIFIC']['DPATH_COVER']
dpath_stego = config['SYSTEM_SPECIFIC']['DPATH_STEGO']
dpath_load_ckpt = config['SYSTEM_SPECIFIC']['DPATH_LOAD_CKPT']
fpath_load_ckpt = ""
if 'FPATH_LOAD_CKPT' in config['SYSTEM_SPECIFIC']:
    fpath_load_ckpt = config['SYSTEM_SPECIFIC']['FPATH_LOAD_CKPT']
dpath_test = config['SYSTEM_SPECIFIC']['DPATH_TEST'] 

dname_stego = os.path.basename(dpath_stego)
pat_bps = re.compile("bitrate-(\d\.*\d*)")
mat_bps = pat_bps.search(dname_stego)
if not mat_bps:
    raise ValueError("Failed to identify BPS information.")
BPS = mat_bps.group(1)

device = torch.device("cuda:0")
dtype_dset = np.float32
dtype = torch.float32


def test_with_cover_stego_biased_proportions(prob_cover=0.99,
                                             prob_stego=0.01):
    print("[Testing with prob_stego: %.3f"%(prob_stego))
    # Create dataset and dataloader
    dset_test = TimitTestSet(dpath_cover,
                             dpath_stego,
                             seed=SEED,
                             prob_cover=prob_cover,
                             prob_stego=prob_stego,
                             dtype=np.float32)    
    n_data = len(dset_test)    
    n_train = math.floor(0.8 * n_data)
    ix_end_valid = n_train
    indices = np.arange(n_data)
    sampler_test = SubsetRandomSampler(indices[ix_end_valid:])
    
    # Create dataloader_train
    dataloader_test = DataLoader(dset_test,
                                 batch_size=BATCH_SIZE,
                                 sampler=sampler_test,
                                 num_workers=N_WORKERS,
                                 pin_memory=True)
    
    # Create model
    model = ModelFactory.create(config)
    model.to(device, dtype=dtype)
    model = nn.DataParallel(model)
    
    global fpath_load_ckpt
    if dpath_load_ckpt:
        if not fpath_load_ckpt:
            fpath_load_ckpt = get_ckpt(dpath_load_ckpt, policy=LOAD_POLICY)
            
        load_model(fpath_load_ckpt, model)
        print("[%s]"%LOAD_POLICY.upper(), fpath_load_ckpt, "has been loaded...") 
    elif fpath_load_ckpt:
        load_model(fpath_load_ckpt, model)    
    
    loss_ce = nn.CrossEntropyLoss()   
    def classification_loss(logits, target_labels):
        return loss_ce(logits, target_labels)

    
    def classify(model, batch):
        audios, labels = batch                
        audios = audios.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(audios)    
        loss = classification_loss(logits, labels)
        acc = accuracy_top1(logits, labels)
        ps = labels.to(torch.float32).mean().item()
        
        return (logits,
                loss.item(),
                acc,
                ps)        
        
    def compute_rates(labels_pred, labels_true):
        # Calculate confusion matrix        
        cm = confusion_matrix(labels_pred, labels_true, labels=(1, 0))
        
        tp = cm[0, 0]  # True stegos  (stegos)
        tn = cm[1, 1]  # True covers  (covers)
        fp = cm[0, 1]  # False stegos (covers)
        fn = cm[1, 0]  # False covers (stegos)
        
        p = tp + fn
        n = tn + fp
        tpr = tp / p  # Sensitivity
        fpr = fp / n  # False alarm (1 - specificity)
        tnr = tn / n  # Specificity
        fnr = fn / p  # Miss rate
        
        return tpr, fpr, tnr, fnr

    # Lists for statistics
    list_stats = []
    list_acc = []
    list_loss = []
    list_prob = []
    
    
    # Lists for true labels, scores, predictions
    list_scores = []
    list_labels = []
    
    num_audios = 0 
    model.eval()   
    
    for epoch in tqdm.tqdm(range(N_REPEATS)):
        # Testing model    
        sum_acc = 0        
        sum_loss = 0
        sum_prob_stego = 0
        list_single_test_preds = []
        list_single_test_labels = []

        for step, batch in enumerate(dataloader_test):
            num_audios += 2*len(batch)        
            with torch.no_grad():   
                # ps denotes prob. of fetching stegos                
                logits, loss, acc, ps = classify(model, batch)            
                sum_acc += acc
                sum_loss += loss
                sum_prob_stego += ps
                
                # Compute score for roc_curve        
                sm = torch.softmax(logits, dim=0)
                list_scores.append(sm[:, 1].cpu().numpy())
                
                _, labels = batch                
                list_single_test_labels.append(labels.cpu().numpy())
                
                preds = logits.topk(1).indices.view(-1) # Predictions
                list_single_test_preds.append(preds.cpu().numpy())
                
        # end of for
        
        avg_acc = sum_acc / len(dataloader_test)
        avg_loss = sum_loss / len(dataloader_test)    
        avg_prob_stego = sum_prob_stego / len(dataloader_test)
        
        # Compute the rates
        labels_pred = np.concatenate(list_single_test_preds)
        labels_true = np.concatenate(list_single_test_labels)
        tpr, fpr, tnr, fnr = compute_rates(labels_pred, labels_true)
        
        fstr = "- Acc:%.4f, Loss:%.6f, Ps:%.4f, " \
               "FA(fpr):%.4f, MD(fnr):%.4f, PE:%.4f"
        print()
        print(fstr%(avg_acc, avg_loss, avg_prob_stego,
                    fpr, fnr, 0.5*(fpr+fnr)))        
        # end of for
        list_acc.append(avg_acc)
        list_loss.append(avg_loss)
        list_prob.append(avg_prob_stego)
        list_labels.append(labels_true)
        list_stats.append({"test_avg_acc": avg_acc,
                           "test_avg_loss": avg_loss,
                           "test_avg_prob_stego": avg_prob_stego,
                           "test_avg_prob_cover": 1 - avg_prob_stego,
                           "test_tpr": tpr,
                           "test_fpr": fpr,
                           "test_tnr": tnr,
                           "test_fnr": fnr,})
    # end of for
       

    
    # Compute ROC
    labels_true = np.concatenate(list_labels)
    y_score = np.concatenate(list_scores)
    roc_fpr, roc_tpr, roc_thr = roc_curve(labels_true, y_score)
    roc_auc = roc_auc_score(labels_true, y_score)
    
    print()
    print("- Avg. acc:", "%.4f ± %.4f"%(np.mean(list_acc), np.std(list_acc)))
    print("- Avg. loss:", "%.6f ± %.4f"%(np.mean(list_loss), np.std(list_loss)))
    print("- Avg. prob:", "%.4f ± %.4f"%(np.mean(list_prob), np.std(list_prob)))
    print("- Total num. tested audios:", num_audios)
    print()
        
    df_stats = pd.DataFrame(list_stats)
    
    
    dict_stats = {"model": MODEL,
                  "steganography": STEGANOGRAPHY.lower(),
                  "ps":ps,
                  "stats": df_stats,
                  "roc": {"roc_auc": roc_auc,
                          "roc_tpr": roc_tpr,
                          "roc_fpr": roc_fpr,
                          "roc_thr": roc_thr}}
    
    return dict_stats
    
if __name__ == "__main__":

    os.makedirs(dpath_test, exist_ok=True)

    stats_eq = test_with_cover_stego_biased_proportions(0.5, 0.5)
    stats_bi = test_with_cover_stego_biased_proportions(0.99, 0.01)

    dict_test = {"stats_eq": stats_eq,
                 "stats_bi": stats_bi}
        
    dname = None
    if fpath_load_ckpt:
        dname = os.path.basename(os.path.dirname(fpath_load_ckpt))
    elif dpath_load_ckpt:
        dname = os.path.basename(dpath_load_ckpt)
    
    fname = "test_" + dname + ".dat"    
    with open(pjoin(dpath_test, fname), "wb") as fout:
        pickle.dump(dict_test, fout)
