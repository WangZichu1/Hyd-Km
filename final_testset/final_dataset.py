import random
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import StandardScaler
from utils import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_full_data_std(full_data_path,batch_size,is_train=True,scaler=None):

    data = th.load(full_data_path,weights_only=False)

    labels = data["test_labels"]
    wfeats = data['test_wfeats']
    rpfeats = data['test_pfeats']
    rsfeats = data['test_sfeats']
    fpfeats = data['test_fpfeats']
    fsfeats = data['test_fsfeats']
    
    wpfeats = th.cat([wfeats,rpfeats], axis=1)
    wsfeats = th.cat([wfeats,rsfeats], axis=1)
    psfeats = th.cat([fpfeats,fsfeats], axis=1)
    #cplx_feats = th.cat([wfeats,fpfeats,fsfeats], axis=1)

    dataset = TensorDataset(psfeats,labels)
    
    if is_train:

        train_feats = th.stack([dataset[i][0] for i in range(len(dataset))])
        train_labels = th.stack([dataset[i][1] for i in range(len(dataset))])


        feat_scaler = StandardScaler()
        train_feats_np = train_feats.numpy()
        train_feats_scaled_np = feat_scaler.fit_transform(train_feats_np)
        train_feats_scaled = th.from_numpy(train_feats_scaled_np).float()


        label_scaler = StandardScaler()
        train_labels_np = train_labels.numpy().reshape(-1, 1)
        train_labels_scaled_np = label_scaler.fit_transform(train_labels_np)
        train_labels_scaled = th.from_numpy(train_labels_scaled_np).float().view(-1, 1)


        dataset = TensorDataset(train_feats_scaled, train_labels_scaled)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            #worker_init_fn=seed_worker,
                            #generator=g,
                            drop_last=False)
        
        return train_loader,dataset,feat_scaler,label_scaler

    else:

        feats = th.stack([dataset[i][0] for i in range(len(dataset))])
        labels = th.stack([dataset[i][1] for i in range(len(dataset))])
        

        feats_np = feats.numpy()
        feats_scaled_np = scaler['feat'].transform(feats_np)
        feats_scaled = th.from_numpy(feats_scaled_np).float()
        
        # 标准化标签（仅限回归任务）
        labels_np = labels.numpy().reshape(-1, 1)
        labels_scaled_np = scaler['label'].transform(labels_np)
        labels_scaled = th.from_numpy(labels_scaled_np).float().view(-1, 1)
        
        test_dataset = TensorDataset(feats_scaled, labels_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                  shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        return test_loader,dataset