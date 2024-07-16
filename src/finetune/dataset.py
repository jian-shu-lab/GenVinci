from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle
from PIL import Image
import glob
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
import open_clip
from transformers import BertForSequenceClassification
import tqdm
# from transformers import AutoProcessor

def identity(x):
    return x

def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap')

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP
    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)
    v_split = normalize(v_split)
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' == 'npy'# type: ignore

def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
    img_label = []
    wind = []
    desc = []
    for _, v in class_index.items():
        n_list = []
        for n in v[:-1]:
            n_list.append(int(n[1:]))
        wind.append(n_list)
        desc.append(v[-1])

    naive_label = {} if naive_label_set is None else naive_label_set
    for _, file in enumerate(img_filename):
        name = int(file[0].split('.')[0])
        naive_label[name] = []
        nl = list(naive_label.keys()).index(name)
        for c, (w, d) in enumerate(zip(wind, desc)):
            if name in w:
                img_label.append((c, d, nl))
                break
    return img_label, naive_label

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

EEG_EXTENSIONS = [
    '.mat'
]

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in EEG_EXTENSIONS)

def make_dataset(dir):

    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, topdown=False)):#
        for fname in fnames:
            if is_mat_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

from PIL import Image
import numpy as np
 

def create_spatial_dataset(dataset, tokenizer_path, test_ratio, image_transform=None):
    
    adata = sc.read_h5ad(os.path.join('dataset', dataset, 'demo.h5ad'))

    if dataset == 'he_st':
        val_samples = ['BC23377'] 
        val_index = [idx.split('_')[0] in val_samples for idx in adata.obs_names]
        val_data = adata[val_index, :].copy()
        train_data = adata[~np.array(val_index), :].copy()
        test_data = val_data.copy()

    if isinstance(image_transform, list): # different transforms for train and test
        train_data = SpatialDataset(dataset, tokenizer_path, train_data, image_transform[0])
        val_data = SpatialDataset(dataset, tokenizer_path, val_data, image_transform[1])
        test_data = SpatialDataset(dataset, tokenizer_path, test_data, image_transform[1])

    else:
        train_data = SpatialDataset(dataset, tokenizer_path, train_data, image_transform)
        val_data = SpatialDataset(dataset, tokenizer_path, val_data, image_transform)
        test_data = SpatialDataset(dataset, tokenizer_path, test_data, image_transform)

    return (train_data, val_data, test_data)

def tokenizer(path, adata): # tokenizer for geneformer

    with open(path + '/gene_median_dictionary.pkl', "rb") as f:
            gene_median_dict = pickle.load(f) # 25424 genes 'ENSG00000000003': 2.001186019549122
    with open(path + '/token_dictionary.pkl', "rb") as f: # 25426 tokens, with pad and mask, '<pad>': 0, '<mask>': 1
        gene_token_dict = pickle.load(f)
    
    gene_dict = dict(zip(list(gene_median_dict.keys()), [True] * len(list(gene_median_dict.keys())))) # 25424 'ENSG00000000003': True 
    coding_miRNA_loc = np.where([gene_dict.get(i, False) for i in adata.var["ensg_id"]])[0] # 15593 get coding miRNA indices in adata.var["ensg_id"]
    norm_factor_vector = np.array([gene_median_dict[i] for i in adata.var["ensg_id"][coding_miRNA_loc]]) # get median for each coding miRNA in adata.var["ensg_id"]
    
    coding_miRNA_ids = adata.var["ensg_id"][coding_miRNA_loc]
    coding_miRNA_tokens = np.array([gene_token_dict[i] for i in coding_miRNA_ids]) # get token ids for genes

    adata = adata[:, coding_miRNA_loc] # select only coding miRNA genes

    # adata = adata[adata.X.sum(axis=1) > 0, :]

    count = adata.X / adata.X.sum(axis=1,keepdims=True) * 10000 / norm_factor_vector # 30536 × 15593

    cell_tokens = []
    # cell_attention_masks = []

    for i in range(len(count)): # select nonzero genes and rank by counts
        nonzero_mask = np.nonzero(count[i])[0] # get nonzero gene indices
        sorted_indices = np.argsort(-count[i][nonzero_mask]) # from large to small rank by counts, get indices
        cell = coding_miRNA_tokens[nonzero_mask][sorted_indices]  
        # cell_mask = [1] * len(cell)

        if len(cell) < 2048:
            cell = np.pad(cell, (0, 2048 - len(cell)), "constant") # pad zero
            # cell_mask = np.pad(cell_mask, (0, 2048 - len(cell_mask)), "constant") 
        else:
            cell = cell[0:2048] # select top 2048 nonzero genes
            # cell_mask = cell_mask[0:2048]

        cell_tokens.append(cell)
        # cell_attention_masks.append(cell_mask)

    cell_tokens = np.array(cell_tokens)
    # cell_attention_masks = np.array(cell_attention_masks)

    # return cell_tokens, cell_attention_masks

    adata.obsm['token'] = cell_tokens
    
    return adata

class SpatialDataset(Dataset):
    
    # Constructor
    def __init__(self, dataset=None, tokenizer_path=None, adata=None, image_transform=None):
        
        # preprocessing for geneformer
        self.adata = tokenizer(tokenizer_path, adata)
        self.tokens = self.adata.obsm['token']

        self.names = self.adata.obs_names.values

        if dataset == 'he_st':
            self.image_path = [os.path.join('dataset', dataset, 'spot_224', self.adata.obs_names[i]) + '.jpg' for i in range(len(self.adata))]
        
        else:                                                  
            raise NotImplementedError

        
        self.image_transform = image_transform

        model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device='cpu')
        # model, preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K', cache_dir='./pretrains/models')
        # self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14") 
        # https://huggingface.co/openai/clip-vit-large-patch14, train with the image size of 224

        # # for debugging, subsampling
        # self.image_path = self.image_path[0:50]
        # self.tokens = self.tokens[0:50]

    # Get size
    def __len__(self):
        return len(self.image_path)
        # return 10
        
    # Get item
    def __getitem__(self, i):

        image_raw = Image.open(self.image_path[i]).convert("RGB") # PIL.Image.Image image mode=RGB size=224x224

        image_norm = self.image_transform(image_raw) # to tensor (512, 512, 3) 

        image_clip = self.preprocess(Image.open(self.image_path[i]))

        count = torch.tensor(self.tokens[i]).long() # torch.Size([2048])

        # att_mask = torch.tensor(self.att_masks[i])

        label = torch.tensor(0).long() # for CLS, if appliable

        return {'image': image_norm, 'count': count, 'label': label, 'image_clip': image_clip, 'id': self.names[i]}


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img
def normalize2(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img
def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


if __name__ == '__main__':
    import scipy.io as scio
    import copy
    import shutil

