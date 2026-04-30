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

    # fpfeats = data['train_fpfeats']
    # fsfeats = data['train_fsfeats']
    # wfeats = data['train_wfeats']
    # #shuffled_wfeats = wfeats[torch.randperm(wfeats.size(0))]
    # rpfeats = data['train_pfeats']
    # rsfeats = data['train_sfeats']
    # labels = data["train_labels"]

    fpfeats = data['test_fpfeats']
    fsfeats = data['test_fsfeats']
    wfeats = data['test_wfeats']
    rpfeats = data['test_pfeats']
    rsfeats = data['test_sfeats']
    labels = data["test_labels"]

    dataset = TensorDataset(wfeats,fpfeats,fsfeats,labels)
    
    if is_train:

        train_wfeats = th.stack([dataset[i][0] for i in range(len(dataset))])
        train_pfeats = th.stack([dataset[i][1] for i in range(len(dataset))])
        train_sfeats = th.stack([dataset[i][2] for i in range(len(dataset))])
        train_labels = th.stack([dataset[i][3] for i in range(len(dataset))])


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
        train_labels_scaled = th.from_numpy(train_labels_scaled_np).float().view(-1, 1)


        dataset = TensorDataset(train_wfeats_scaled, train_pfeats_scaled,train_sfeats_scaled,train_labels_scaled)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            #worker_init_fn=seed_worker,
                            #generator=g,
                            drop_last=False)
        
        return train_loader,dataset,wfeat_scaler,pfeat_scaler,sfeat_scaler,label_scaler

    else:

        wfeats = th.stack([dataset[i][0] for i in range(len(dataset))])
        pfeats = th.stack([dataset[i][1] for i in range(len(dataset))])
        sfeats = th.stack([dataset[i][2] for i in range(len(dataset))])
        labels = th.stack([dataset[i][3] for i in range(len(dataset))])
        

        wfeats_np = wfeats.numpy()
        wfeats_scaled_np = scaler['wfeat'].transform(wfeats_np)
        wfeats_scaled = th.from_numpy(wfeats_scaled_np).float()

        pfeats_np = pfeats.numpy()
        pfeats_scaled_np = scaler['pfeat'].transform(pfeats_np)
        pfeats_scaled = th.from_numpy(pfeats_scaled_np).float()

        sfeats_np = sfeats.numpy()
        sfeats_scaled_np = scaler['sfeat'].transform(sfeats_np)
        sfeats_scaled = th.from_numpy(sfeats_scaled_np).float()
        

        labels_np = labels.numpy().reshape(-1, 1)
        labels_scaled_np = scaler['label'].transform(labels_np)
        labels_scaled = th.from_numpy(labels_scaled_np).float().view(-1, 1)
        
        test_dataset = TensorDataset(wfeats_scaled, pfeats_scaled,sfeats_scaled,labels_scaled)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                  shuffle=False,
                                  #worker_init_fn=seed_worker,
                                  #generator=g,
                                  drop_last=False)
        return test_loader,dataset