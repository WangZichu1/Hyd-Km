import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
# seed = 42
# b_s = 64
pth_path = "path/esm_subdataset.pth"
full_data_path = 'path/full_data.pth'
idx_path = ""

def loaddata_std(pth_path,full_data_path,batch_size,is_train=True, scaler=None):

    waterdata = torch.load(full_data_path,weights_only=False)
    cached_data = torch.load(pth_path,weights_only=False)
    
    feats,subfeats, labels = cached_data["profeats"],cached_data['subfeats'], cached_data["labels"]
    wfeats = waterdata['waterfeats']
    #wlabels = waterdata["labels"]
    
    cplx_feats = torch.cat([feats, subfeats], axis=1)
    labels = torch.log10(labels)
    
    dataset = TensorDataset(subfeats, labels)

    total_samples = len(dataset)
    train_samples = int(0.8 * total_samples)
    valid_samples = total_samples - train_samples
    train_dataset, valid_dataset = random_split(dataset, [train_samples, valid_samples])

    # 
    if is_train:
        # 
        train_feats = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_labels = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])

        # 
        feat_scaler = StandardScaler()
        train_feats_np = train_feats.numpy()
        train_feats_scaled_np = feat_scaler.fit_transform(train_feats_np)
        train_feats_scaled = torch.from_numpy(train_feats_scaled_np).float()

        # 
        label_scaler = StandardScaler()
        train_labels_np = train_labels.numpy().reshape(-1, 1)
        train_labels_scaled_np = label_scaler.fit_transform(train_labels_np)
        train_labels_scaled = torch.from_numpy(train_labels_scaled_np).float().view(-1, 1)

        # 
        train_dataset = TensorDataset(train_feats_scaled, train_labels_scaled)
        
        # 
        valid_feats = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])
        valid_labels = torch.stack([valid_dataset[i][1] for i in range(len(valid_dataset))])

        valid_feats_np = valid_feats.numpy()
        valid_feats_scaled_np = feat_scaler.transform(valid_feats_np)
        valid_feats_scaled = torch.from_numpy(valid_feats_scaled_np).float()

        valid_labels_np = valid_labels.numpy().reshape(-1, 1)
        valid_labels_scaled_np = label_scaler.transform(valid_labels_np)
        valid_labels_scaled = torch.from_numpy(valid_labels_scaled_np).float().view(-1, 1)
        
        valid_dataset = TensorDataset(valid_feats_scaled, valid_labels_scaled)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        # 
        return train_loader,valid_loader,train_dataset, valid_dataset,feat_scaler, label_scaler
    
    else:
        # 
        valid_feats = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])
        valid_labels = torch.stack([valid_dataset[i][1] for i in range(len(valid_dataset))])
        
        # 
        valid_feats_np = valid_feats.numpy()
        valid_feats_scaled_np = scaler['feat'].transform(valid_feats_np)
        valid_feats_scaled = torch.from_numpy(valid_feats_scaled_np).float()
        
        # 
        valid_labels_np = valid_labels.numpy().reshape(-1, 1)
        valid_labels_scaled_np = scaler['label'].transform(valid_labels_np)
        valid_labels_scaled = torch.from_numpy(valid_labels_scaled_np).float().view(-1, 1)
        
        valid_dataset = TensorDataset(valid_feats_scaled, valid_labels_scaled)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, 
                                  shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        return valid_loader