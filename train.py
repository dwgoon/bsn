import os
from os.path import join as pjoin
import platform
import random
import math

import numpy as np
import pandas as pd

import torch
from torch import nn

from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from models import ModelFactory
import utils
from utils import load_model, save_model
from utils import accuracy_top1
from utils import visdom_create_lineplot_window
from utils import visdom_clear_environments
from utils import get_ckpt
from dataset import TimitTrainSet, TimitTestSet

import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse configruations for training deep learning models.')
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
    MODEL = config['MODEL'].lower()
    SEED = int(config['SEED'])
    N_EPOCHS = config['TRAIN']['N_EPOCHS']
    LEARNING_RATE = config['TRAIN']['LEARNING_RATE']
    FREQ_EPOCH_CKPT = config['TRAIN']['FREQ_EPOCH_CKPT']
    FREQ_EPOCH_VALID = config['TRAIN']['FREQ_EPOCH_VALID']
    FREQ_EPOCH_TEST = config['TRAIN']['FREQ_EPOCH_TEST']
    CLIP_GRAD_VALUE = config['TRAIN']['CLIP_GRAD_VALUE']
    USE_NORM = config['TRAIN']['USE_NORM']
    LOAD_POLICY = config['TRAIN']['LOAD_POLICY']
    
    # Augmentation
    CROP_SIZE = int(config['TRAIN']['CROP_SIZE'])
    APPLY_AUGMENTATION = config['TRAIN']['APPLY_AUGMENTATION']
    
    config_augmentation = config['TRAIN']['AUGMENTATION']
    PROB_CUTOUT = float(config_augmentation['PROB_CUTOUT'])
    PROB_FLIP_LR = float(config_augmentation['PROB_FLIP_LR'])
    PROB_FLIP_SIGN = float(config_augmentation['PROB_FLIP_SIGN'])
    PROB_SHUFFLE_SEGMENTS = float(config_augmentation['PROB_SHUFFLE_SEGMENTS'])
    PROB_ADD_NOISE = float(config_augmentation['PROB_ADD_NOISE'])
    NUM_SEGMENTS = int(config_augmentation['NUM_SEGMENTS'])
    CUTOUT_RATIO = float(config_augmentation['CUTOUT_RATIO'])

    # System specific options
    BATCH_SIZE = config['SYSTEM_SPECIFIC']['BATCH_SIZE']
    N_WORKERS = config['SYSTEM_SPECIFIC']['N_WORKERS']
    dpath_cover = config['SYSTEM_SPECIFIC']['DPATH_COVER']
    dpath_stego = config['SYSTEM_SPECIFIC']['DPATH_STEGO']
    dname_stego = os.path.basename(dpath_stego)
    
    DPATH_SAVE_CKPT = config['SYSTEM_SPECIFIC']['DPATH_SAVE_CKPT'] 
    DPATH_LOAD_CKPT = config['SYSTEM_SPECIFIC']['DPATH_LOAD_CKPT'] 
    if 'FPATH_LOAD_CKPT' in  config['SYSTEM_SPECIFIC']:
        fpath_load_ckpt = config['SYSTEM_SPECIFIC']['FPATH_LOAD_CKPT']
    else:
        fpath_load_ckpt = ""
    
    USE_TENSORBOARD = config['SYSTEM_SPECIFIC']['USE_TENSORBOARD']
    tb_writer = None
    if USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
 
    vis = None
    USE_VISDOM = config['SYSTEM_SPECIFIC']['USE_VISDOM']
    VISDOM_CLEAR_ENVS = config['SYSTEM_SPECIFIC']['VISDOM_CLEAR_ENVS']
    if USE_VISDOM:
        import visdom
        str_time_now = utils.get_str_now()
        str_env = "{}_{}".format(MODEL.upper(), str_time_now)
        vis = visdom.Visdom(env=str_env)
        
        if VISDOM_CLEAR_ENVS:
            visdom_clear_environments(vis)
        
        viswin_loss_train = visdom_create_lineplot_window(
                              vis,                              
                              xlabel='Epoch',
                              ylabel='Avg. Loss', 
                              title='Training Loss')

        viswin_loss_valid = visdom_create_lineplot_window(
                              vis,                              
                              xlabel='Epoch',
                              ylabel='Avg. Loss', 
                              title='Validation Loss')
        
        viswin_loss_test = visdom_create_lineplot_window(
                              vis,                              
                              xlabel='Epoch',
                              ylabel='Avg. Loss', 
                              title='Test Loss')
        
        viswin_acc_train = visdom_create_lineplot_window(
                              vis,
                              xlabel='Epoch',
                              ylabel='Avg. Accuracy', 
                              title='Training Accuracy')

        viswin_acc_valid = visdom_create_lineplot_window(
                              vis,
                              xlabel='Epoch',
                              ylabel='Avg. Accuracy', 
                              title='Validation Accuracy')
        
        viswin_acc_test = visdom_create_lineplot_window(
                              vis,
                              xlabel='Epoch',
                              ylabel='Avg. Accuracy', 
                              title='Test Accuracy')
    # end of setting visdom

    # Set device and data types
    device = torch.device("cuda:0")
    dtype_dset = np.float32
    dtype = torch.float32

    # Create dataset and dataloader
    dset_train = TimitTrainSet(dpath_cover, dpath_stego, 
                            seed=SEED,
                            dtype=dtype_dset,
                            use_librosa=USE_NORM,
                            random_subset=True,
                            crop_size=CROP_SIZE,
                            augmentation=APPLY_AUGMENTATION,
                            prob_cutout=PROB_CUTOUT,
                            prob_flip_lr=PROB_FLIP_LR,
                            prob_flip_sign=PROB_FLIP_SIGN,
                            prob_shuffle_segments=PROB_SHUFFLE_SEGMENTS,
                            prob_add_noise=PROB_ADD_NOISE,
                            num_segments=NUM_SEGMENTS,
                            cutout_ratio=CUTOUT_RATIO)
    dset_valid = TimitTrainSet(dpath_cover, dpath_stego,
                               seed=SEED,
                               dtype=dtype_dset,
                               use_librosa=USE_NORM,
                               random_subset=True,
                               crop_size=16000)
    dset_test = TimitTestSet(dpath_cover,
                             dpath_stego,
                             seed=SEED,
                             prob_cover=0.5,
                             prob_stego=0.5,
                             dtype=dtype_dset,
                             use_librosa=USE_NORM)

    n_data = len(dset_train)
    print("- Num. total audios for training:", n_data)

    n_train = math.floor(0.8 * n_data)
    ix_end_train = math.floor(0.75 * n_train)
    ix_end_valid = n_train
    indices = np.arange(n_data)
    sampler_train = SubsetRandomSampler(indices[:ix_end_train])
    sampler_valid = SubsetRandomSampler(indices[ix_end_train:ix_end_valid])
    sampler_test = SubsetRandomSampler(indices[ix_end_valid:])

    # Create dataloaders
    dataloader_train = DataLoader(dset_train,
                                  batch_size=BATCH_SIZE,
                                  sampler=sampler_train,
                                  num_workers=N_WORKERS,
                                  pin_memory=True)

    dataloader_valid = DataLoader(dset_valid,
                                  batch_size=BATCH_SIZE,
                                  sampler=sampler_valid,
                                  num_workers=N_WORKERS,
                                  pin_memory=True)

    dataloader_test = DataLoader(dset_test,
                                 batch_size=BATCH_SIZE,
                                 sampler=sampler_test,
                                 num_workers=N_WORKERS,
                                 pin_memory=True)

    # Create model    
    model = ModelFactory.create(config)    
    model.to(device, dtype=dtype)
    lr = float(LEARNING_RATE)
    optimizer = model.get_optimizer(model, lr)
    lr_scheduler = model.get_lr_scheduler(optimizer)       

    if DPATH_LOAD_CKPT:
        if not fpath_load_ckpt:
            fpath_load_ckpt = get_ckpt(DPATH_LOAD_CKPT, LOAD_POLICY) #get_best_ckpt_with_criterion(dpath_load_ckpt, LOAD_POLICY)
        load_model(fpath_load_ckpt, model)
        print("[%s]"%(LOAD_POLICY.upper()), fpath_load_ckpt, "has been loaded...")
    # end of if
    model = nn.DataParallel(model)

    loss_ce = nn.CrossEntropyLoss()   
    def classification_loss(logits, target_labels):
        return loss_ce(logits, target_labels)

    def classify(model, batch):
        covers, stegos = batch
        
        n_half = covers.shape[0]
        
        # Shuffle indices
        ind_base = np.arange(0, 2*n_half, 2, dtype=np.int32)
        offset_cover = np.random.randint(0, 2, n_half)
        offset_stego = offset_cover ^ 1
        ind_cover = ind_base + offset_cover
        ind_stego = ind_base + offset_stego
        
        x = torch.zeros((2*n_half, *covers.shape[1:]),
                        dtype=covers.dtype,
                        device=device)

        covers = covers.to(device)        
        stegos = stegos.to(device)
        
        x[ind_cover] = covers
        x[ind_stego] = stegos

        target_labels = torch.zeros(2*n_half,
                                    dtype=torch.long,
                                    device=device)
        target_labels[ind_stego] = 1              

        logits = model(x)
        loss = classification_loss(logits, target_labels)
        acc = accuracy_top1(logits, target_labels)
        return loss, acc
    # end of def

    def classify_test(model, batch):
        audios, labels = batch                
        audios = audios.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    
        logits = model(audios)    
        loss = classification_loss(logits, labels)
        acc = accuracy_top1(logits, labels)
        return loss.item(), acc
    # end of def
    
    #print("[Model specification]")
    #rint(model)
    
    # Open a CSV file for recording statistics.    
    os.makedirs(DPATH_SAVE_CKPT, exist_ok=True)
    fout_stats = open(pjoin(DPATH_SAVE_CKPT, 'stats.csv'), "w+") 
    fout_stats.write("epoch," \
                     "train_avg_loss,train_avg_acc," \
                     "valid_avg_loss,valid_avg_acc," \
                     "test_avg_loss,test_avg_acc\n")    
    
    max_acc_valid = 0
    min_loss_valid = np.finfo(np.float16).max
    
    for epoch in range(1, N_EPOCHS+1):
        stats = {'epoch': epoch}
        list_loss = []
        list_acc = []
        
        # Training
        model.train()
        if lr_scheduler:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = lr_scheduler.get_lr()
        
        for step, batch in enumerate(dataloader_train):
            optimizer.zero_grad()        
            loss, acc = classify(model, batch)                    
            loss.backward()
            if CLIP_GRAD_VALUE > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_VALUE)
            optimizer.step()        
            print('[EPOCH:{}/{}][STEP:{}][LR:{}]'.format(epoch, N_EPOCHS, step, lr))
    
            list_loss.append(loss.item())
            list_acc.append(acc)
            print("\tBatch-Loss:", loss.item(), "Batch-Acc.:", acc)
            print("\tAvg-Loss:", np.mean(list_loss), "Avg-Acc:", np.mean(list_acc))
        # end of for
        print()
        
        avg_loss_train = np.mean(list_loss)
        avg_acc_train = np.mean(list_acc)
    
        stats['train_avg_loss'] = avg_loss_train
        stats['train_avg_acc'] = avg_acc_train
    
        if lr_scheduler:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(avg_loss_train)
            else:
                lr_scheduler.step()
     
        
        # Validating
        stats['valid_avg_loss'] = np.nan
        stats['valid_avg_acc'] = np.nan
    
        if epoch % FREQ_EPOCH_VALID == 0:
            model.eval()
            sum_acc = 0        
            sum_loss = 0
            for step, batch in enumerate(dataloader_valid):
                
                with torch.no_grad():            
                    loss, acc = classify(model, batch)
                    sum_acc += acc
                    sum_loss += loss.item()
            # end of for
            
            avg_acc_valid = sum_acc / len(dataloader_valid)
            avg_loss_valid = sum_loss / len(dataloader_valid)
    
            stats['valid_avg_loss'] = avg_loss_valid
            stats['valid_avg_acc'] = avg_acc_valid
    
            print("[Validation] acc: {}, loss: {}".format(avg_acc_valid, avg_loss_valid))
            print()

        # end of Validating
    
        # Testing    
        stats['test_avg_loss'] = np.nan
        stats['test_avg_acc'] = np.nan
    
        if epoch % FREQ_EPOCH_TEST == 0:
            model.eval()
            sum_acc = 0
            sum_loss = 0
            for step, batch in enumerate(dataloader_test):
                with torch.no_grad():
                    loss, acc = classify_test(model, batch)
                    sum_acc += acc
                    sum_loss += loss
            # end of for
    
            avg_acc_test = sum_acc / len(dataloader_test)
            avg_loss_test = sum_loss / len(dataloader_test)
    
            stats['test_avg_loss'] = avg_loss_test
            stats['test_avg_acc'] = avg_acc_test
    
            print("[Test] acc: {}, loss: {}".format(avg_acc_test, avg_loss_test))
            print()    
        # end of Testing

        # Write stats in file.            
        fout_stats.write("%d,%f,%f,%f,%f,%f,%f\n"%(stats["epoch"],
                                                   stats["train_avg_loss"],
                                                   stats["train_avg_acc"],
                                                   stats["valid_avg_loss"],
                                                   stats["valid_avg_acc"],
                                                   stats["test_avg_loss"],
                                                   stats["test_avg_acc"]))
        fout_stats.flush()
    
        if (epoch % FREQ_EPOCH_VALID == 0) and (epoch % FREQ_EPOCH_CKPT == 0):            
            fstr_fname = "ckpt_max-epoch_{}_{}.pth.tar"
            fname = fstr_fname.format(dname_stego, MODEL)
            fpath_ckpt = save_model(model,                                    
                                    fname=fname,
                                    dpath=DPATH_SAVE_CKPT)
            
            print('[SAVE CKPT for MAX-EPOCH] {}'.format(fpath_ckpt))
            
            if avg_acc_valid > max_acc_valid:
                fstr_fname = "ckpt_max-acc_{}_{}.pth.tar"
                fname = fstr_fname.format(dname_stego, MODEL)
                fpath_ckpt = save_model(model,
                                        fname=fname,
                                        dpath=DPATH_SAVE_CKPT)
                print('[SAVE CKPT for MAX-ACC] {}'.format(fpath_ckpt))
                max_acc_valid = avg_acc_valid 
                
            if avg_loss_valid < min_loss_valid:
                fstr_fname = "ckpt_min-loss_{}_{}.pth.tar"
                fname = fstr_fname.format(dname_stego, MODEL)
                fpath_ckpt = save_model(model,
                                        fname=fname,
                                        dpath=DPATH_SAVE_CKPT)
                print('[SAVE CKPT for MIN-LOSS] {}'.format(fpath_ckpt))
                min_loss_valid = avg_loss_valid 
        print()
        # end of if
            
    
        if tb_writer:
            tb_writer.add_scalar('Loss/Train', avg_loss_train, epoch)
            tb_writer.add_scalar('Loss/Valid', avg_loss_valid, epoch)
    
            tb_writer.add_scalar('Accuracy/Train', avg_acc_train, epoch)
            tb_writer.add_scalar('Accuracy/Valid', avg_acc_valid, epoch)
    
            if epoch % FREQ_EPOCH_TEST == 0:                
                tb_writer.add_scalar('Loss/Test', avg_loss_test, epoch)
                tb_writer.add_scalar('Accuracy/Test', avg_acc_test, epoch)
        # end of if tb_writer
        if vis:
            vis.line(X=np.array([epoch]),
                     Y=np.array([avg_loss_train]),
                     win=viswin_loss_train,
                     update='append')
    
            
            vis.line(X=np.array([epoch]),
                     Y=np.array([avg_acc_train]),
                     win=viswin_acc_train,
                     update='append')
    
            if epoch % FREQ_EPOCH_VALID == 0:                                
                vis.line(X=np.array([epoch]),
                         Y=np.array([avg_loss_valid]),
                         win=viswin_loss_valid,
                         update='append')
                
    
                vis.line(X=np.array([epoch]),
                         Y=np.array([avg_acc_valid]),
                         win=viswin_acc_valid,
                         update='append')
                
    
            if epoch % FREQ_EPOCH_TEST == 0:                
                vis.line(X=np.array([epoch]),
                         Y=np.array([avg_loss_test]),
                         win=viswin_loss_test,
                         update='append')
    
                vis.line(X=np.array([epoch]),
                         Y=np.array([avg_acc_test]),
                         win=viswin_acc_test,
                         update='append')
        # end of if vis    
    # end of for epoch

    if tb_writer:
        tb_writer.close()
# end of main
