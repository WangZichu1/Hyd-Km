import random
import torch as th
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import StandardScaler
from utils import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split


def loaddata_std(full_data_path,idx_path,fold_id,batch_size,is_train=True, scaler=None):

    waterdata = torch.load(full_data_path,weights_only=False)
    
    labels = waterdata["train_labels"]
    wfeats = waterdata['train_wfeats']
    #shuffled_wfeats = wfeats[torch.randperm(wfeats.size(0))]#To identify the usage of hydration
    #rpfeats = waterdata['p_re_feats']
    #rsfeats = waterdata['s_re_feats']
    fpfeats = waterdata['train_fpfeats']
    fsfeats = waterdata['train_fsfeats']
    
    #cplx_feats = th.cat([wfeats,rpfeats,rsfeats], axis=1)
    #labels = torch.log10(labels)
    
    folds = th.load(idx_path,weights_only=False)
    train_idx, val_idx = folds[fold_id]

    train_wfeats_fold = wfeats[train_idx]
    train_pfeats_fold = fpfeats[train_idx]
    train_sfeats_fold = fsfeats[train_idx]   
    train_labels_fold = labels[train_idx]
    
    valid_wfeats_fold = wfeats[val_idx]
    valid_pfeats_fold = fpfeats[val_idx]
    valid_sfeats_fold = fsfeats[val_idx]
    valid_labels_fold = labels[val_idx]
###############################################################################################################
    train_dataset = TensorDataset(train_wfeats_fold,train_pfeats_fold,train_sfeats_fold,train_labels_fold)
    valid_dataset = TensorDataset(valid_wfeats_fold,valid_pfeats_fold,valid_sfeats_fold,valid_labels_fold)

    # 
    if is_train:
        
        train_wfeats = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_pfeats = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
        train_sfeats = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])
        train_labels = torch.stack([train_dataset[i][3] for i in range(len(train_dataset))])

        
        wfeat_scaler = StandardScaler()
        train_wfeats_np = train_wfeats.numpy()
        train_wfeats_scaled_np = wfeat_scaler.fit_transform(train_wfeats_np)
        train_wfeats_scaled = torch.from_numpy(train_wfeats_scaled_np).float()

        pfeat_scaler = StandardScaler()
        train_pfeats_np = train_pfeats.numpy()
        train_pfeats_scaled_np = pfeat_scaler.fit_transform(train_pfeats_np)
        train_pfeats_scaled = torch.from_numpy(train_pfeats_scaled_np).float()

        sfeat_scaler = StandardScaler()
        train_sfeats_np = train_sfeats.numpy()
        train_sfeats_scaled_np = sfeat_scaler.fit_transform(train_sfeats_np)
        train_sfeats_scaled = torch.from_numpy(train_sfeats_scaled_np).float()

        
        label_scaler = StandardScaler()
        train_labels_np = train_labels.numpy().reshape(-1, 1)
        train_labels_scaled_np = label_scaler.fit_transform(train_labels_np)
        train_labels_scaled = torch.from_numpy(train_labels_scaled_np).float().view(-1, 1)

        
        train_dataset = TensorDataset(train_wfeats_scaled,train_pfeats_scaled,train_sfeats_scaled, train_labels_scaled)
        
        
        valid_wfeats = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])
        valid_pfeats = torch.stack([valid_dataset[i][1] for i in range(len(valid_dataset))])
        valid_sfeats = torch.stack([valid_dataset[i][2] for i in range(len(valid_dataset))])
        valid_labels = torch.stack([valid_dataset[i][3] for i in range(len(valid_dataset))])

        valid_wfeats_np = valid_wfeats.numpy()
        valid_wfeats_scaled_np = wfeat_scaler.transform(valid_wfeats_np)
        valid_wfeats_scaled = torch.from_numpy(valid_wfeats_scaled_np).float()
        
        valid_pfeats_np = valid_pfeats.numpy()
        valid_pfeats_scaled_np = pfeat_scaler.transform(valid_pfeats_np)
        valid_pfeats_scaled = torch.from_numpy(valid_pfeats_scaled_np).float()
        
        valid_sfeats_np = valid_sfeats.numpy()
        valid_sfeats_scaled_np = sfeat_scaler.transform(valid_sfeats_np)
        valid_sfeats_scaled = torch.from_numpy(valid_sfeats_scaled_np).float()

        valid_labels_np = valid_labels.numpy().reshape(-1, 1)
        valid_labels_scaled_np = label_scaler.transform(valid_labels_np)
        valid_labels_scaled = torch.from_numpy(valid_labels_scaled_np).float().view(-1, 1)
        
        valid_dataset = TensorDataset(valid_wfeats_scaled,valid_pfeats_scaled,valid_sfeats_scaled, valid_labels_scaled)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        
        return train_loader,valid_loader,train_dataset,valid_dataset,wfeat_scaler,pfeat_scaler,sfeat_scaler,label_scaler
    
    else:
        
        valid_wfeats = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])
        valid_pfeats = torch.stack([valid_dataset[i][1] for i in range(len(valid_dataset))])
        valid_sfeats = torch.stack([valid_dataset[i][2] for i in range(len(valid_dataset))])
        valid_labels = torch.stack([valid_dataset[i][3] for i in range(len(valid_dataset))])
        
        
        valid_wfeats_np = valid_wfeats.numpy()
        valid_wfeats_scaled_np = scaler['wfeat'].transform(valid_wfeats_np)
        valid_wfeats_scaled = torch.from_numpy(valid_wfeats_scaled_np).float()

        valid_pfeats_np = valid_pfeats.numpy()
        valid_pfeats_scaled_np = scaler['pfeat'].transform(valid_pfeats_np)
        valid_pfeats_scaled = torch.from_numpy(valid_pfeats_scaled_np).float()

        valid_sfeats_np = valid_sfeats.numpy()
        valid_sfeats_scaled_np = scaler['sfeat'].transform(valid_sfeats_np)
        valid_sfeats_scaled = torch.from_numpy(valid_sfeats_scaled_np).float()
        
        
        valid_labels_np = valid_labels.numpy().reshape(-1, 1)
        valid_labels_scaled_np = scaler['label'].transform(valid_labels_np)
        valid_labels_scaled = torch.from_numpy(valid_labels_scaled_np).float().view(-1, 1)
        
        valid_dataset = TensorDataset(valid_wfeats_scaled,valid_pfeats_scaled,valid_sfeats_scaled,valid_labels_scaled)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                                  shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        return valid_loader