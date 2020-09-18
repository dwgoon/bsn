import os
import gc
import glob
import time
import random
from collections import defaultdict

import numpy as np
from scipy.io import wavfile
import wave
import librosa

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class TimitTrainSet(Dataset):
    
    def __init__(self,
                 dirs_cover,
                 dirs_stego,
                 seed=None,
                 dtype=np.float16,
                 use_librosa=False,
                 random_subset=False,
                 augmentation=False,
                 crop_size=8000,
                 prob_cutout=0.5,
                 prob_flip_lr=0.5,
                 prob_flip_sign=0.5,
                 prob_shuffle_segments=0.5,
                 prob_add_noise=0.5,
                 num_segments=4,
                 cutout_ratio=0.1,
                 transform=None):
        
        self.ext = 'wav'
        self.dirs_cover = self._initialize_dirs(dirs_cover)            
        self.dirs_stego = self._initialize_dirs(dirs_stego)        
        self.seed = seed
        self.transform = transform
        self.dtype = dtype 
        self.use_librosa = use_librosa
        self.random_subset = random_subset
        self.augmentation = augmentation
        self.crop_size = crop_size
        
        self.prob_cutout = prob_cutout
        self.prob_flip_lr = prob_flip_lr
        self.prob_flip_sign = prob_flip_sign
        self.prob_shuffle_segments = prob_shuffle_segments
        self.prob_add_noise = prob_add_noise
        self.num_segments = num_segments
        self.cutout_ratio = cutout_ratio

        if seed: 
            if not isinstance(seed, int):
                raise ValueError("seed should be integer, not {}".format(seed))                    
                
            np.random.seed(seed)
        else:
            np.random.seed()
               
        self.speaches_cover = defaultdict(list)
        self.speaches_stego = defaultdict(list)
        
        
        keys_cover = set()
        keys_stego = set()
        
        for dpath in self.dirs_cover:    
            speaches = self._load_fpaths(dpath)            
            keys_cover |= set(speaches.keys())
            self.speaches_cover.update(speaches)
        # end of for
        
        for dpath in self.dirs_stego:
            speaches = self._load_fpaths(dpath)            
            keys_stego |= set(speaches.keys())
            self.speaches_stego.update(speaches)
        # end of for    
                        
        keys_cover = sorted(keys_cover)
        keys_stego = sorted(keys_stego)        
        self.keys_cover = np.array(keys_cover, dtype=str)
        self.keys_stego = np.array(keys_stego, dtype=str)
               
        # Create labels.
        num_pairs = len(self.keys_cover)
        self.labels_cover = np.zeros((num_pairs,), dtype=np.long)
        self.labels_stego = np.ones((num_pairs,), dtype=np.long)
        
        if seed:  # Shuffle for training
            ind = np.arange(num_pairs)
            np.random.shuffle(ind)
            self.keys_cover = self.keys_cover[ind]
            self.keys_stego = self.keys_stego[ind]

        self.cnts = {}
        for key in self.keys_cover:
            self.cnts[key] = 0
            
        self.cache = {}
        gc.collect()

    def _initialize_dirs(self, dirs):        
        fstr_err_msg = "Wrong type of directories: %s"
        
        if isinstance(dirs, list):
            return dirs
        elif isinstance(dirs, str):
            return [dirs]
        else:
            raise ValueError(fstr_err_msg%(type(dirs)))
        
    
    def _load_fpaths(self, dpath):
        #fpaths = []
        speaches = defaultdict(list)
        fpath_pat = os.path.join(dpath, '*.{}'.format(self.ext))  
        glob_dir = sorted(glob.glob(fpath_pat))
        for fpath in glob_dir:        
            #fpaths.append(fpath)            
            fname = os.path.basename(fpath)
            idx = fname.find("chunk") # the end index of substring
            iden = fname[:idx-1]  # Identity of wav file            
            speaches[iden].append(fpath)
        # end of for
        return speaches
        
    def _load_file(self, fpath):
        if fpath in self.cache:
            return self.cache[fpath]
    
        if self.use_librosa:
            wav = wave.open(fpath, "r")
            sr = wav.getframerate()
            arr, sr = librosa.load(fpath, sr)
        else:
            sr, arr = wavfile.read(fpath)

        if arr.ndim == 2:
            arr = arr.transpose(1, 0)
        elif arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0])
        else:
            raise ValueError("Illegal dimension of audio: %d"%arr.ndim)

        arr = arr.astype(self.dtype)
        self.cache[fpath] = arr
        
        return arr
        
    def __getitem__(self, index):
        cnt = self.cnts[self.keys_cover[index]]
        speaches_cover = self.speaches_cover[self.keys_cover[index]]
        speaches_stego = self.speaches_stego[self.keys_stego[index]]
        
        if self.random_subset:
            ix_selected = np.random.randint(0, len(speaches_cover))        
        else:
            ix_selected = cnt
        self.cnts[self.keys_cover[index]] = (cnt + 1) % len(speaches_cover)
        
        fpath_cover = speaches_cover[ix_selected]
        fpath_stego = speaches_stego[ix_selected]
        
        cover = self._load_file(fpath_cover)
        stego = self._load_file(fpath_stego)
        
        # Randomly crop the audio samples
        len_audio = cover.shape[-1]
        if self.crop_size > 0 and self.crop_size < len_audio:
            # In the random cropping, sizes can be changed.
            ix_crop_beg = random.randint(0, len_audio - self.crop_size)
            ix_crop_end = ix_crop_beg + self.crop_size
            cover = cover[:, ix_crop_beg:ix_crop_end]
            stego = stego[:, ix_crop_beg:ix_crop_end]
        
        if self.augmentation:
            if self.prob_cutout > 0:
                rand_cutout = random.random()            
                if rand_cutout < self.prob_cutout:     
                    len_audio = cover.shape[-1]
                    cutout_size = int(self.cutout_ratio*len_audio)
                    ix_cut_beg = random.randint(0,
                                                len_audio - cutout_size)
                    ix_cut_end = ix_cut_beg + cutout_size                    
                    val = 0 #np.random.randint(cover.min(), cover.max())
                    cover[:, ix_cut_beg:ix_cut_end] = val
                    stego[:, ix_cut_beg:ix_cut_end] = val                                        
            
            if self.prob_flip_lr > 0:
                rand_flip_lr = random.random()            
                if rand_flip_lr < self.prob_flip_lr:
                    cover = np.fliplr(cover)
                    stego = np.fliplr(stego)
                                            
            if self.prob_flip_sign > 0:
                rand_flip_sign = random.random()
                if rand_flip_sign < self.prob_flip_sign:
                    cover = -cover
                    stego = -stego
                                            
            if self.prob_shuffle_segments > 0:
                rand_shuffle_segments = random.random()
                if rand_shuffle_segments < self.prob_shuffle_segments:
                    cover_splitted = np.split(cover, self.num_segments, axis=1)
                    stego_splitted = np.split(stego, self.num_segments, axis=1)

                    # cover_splitted is a list object.
                    ind = np.arange(len(cover_splitted))
                    np.random.shuffle(ind)
                            
                    cover = np.concatenate([cover_splitted[i] for i in ind],
                                           axis=1)
                    stego = np.concatenate([stego_splitted[i] for i in ind],
                                           axis=1)                        
                                    
            if self.prob_add_noise > 0:
                rand_add_noise = random.random()
                if rand_add_noise < self.prob_add_noise:
                    noise = np.random.randint(-1, 2, size=cover.shape)
                    noise = noise.astype(self.dtype)
                    cover = cover + noise
                    stego = stego + noise
            

            cover = cover.copy()
            stego = stego.copy()
        # end of if self.augmentation
        
        if self.transform:
            cover = self.transform(cover)
            stego = self.transform(stego)
            
        return cover, stego            
      
    def __len__(self):
        return len(self.keys_cover)
    
    
class TimitTestSet(TimitTrainSet):
    
    def __init__(self,
                 *args,
                 prob_cover=0.9,
                 prob_stego=0.1,
                 **kwargs):
        
        super(TimitTestSet, self).__init__(*args, **kwargs)
        
        if prob_cover + prob_stego != 1:
            err_msg = "Sum. of prob_cover and prob_stego should be equal to 1."
            raise ValueError(err_msg)
            
        self.prob_cover = prob_cover
        self.prob_stego = prob_stego
        
    def __getitem__(self, index):
        prob = random.random()
        label = None
        if prob <= self.prob_stego:
            speaches = self.speaches_stego[self.keys_stego[index]]
            label = 1
        elif self.prob_stego < prob:
            speaches = self.speaches_cover[self.keys_cover[index]]
            label = 0
        else:
            raise RuntimeError("This point should not be reached.")
        
        ix_selected = np.random.randint(0, len(speaches))        
        fpath = speaches[ix_selected]
        audio = self._load_file(fpath)

        if self.transform:
            audio = self.transform(audio)
            
        return audio, label
      
    def __len__(self):
        return len(self.keys_cover)
