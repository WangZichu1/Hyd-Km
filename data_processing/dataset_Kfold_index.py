import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

file_path = "C:/Users/wangz/Desktop/11-24-water/esm_subdataset.pth"
#seed = 42

data = th.load(file_path,weights_only=False)
#feats,labels,mask,ezy_keys,waterlengths,seqlength,maxlength = data["feats"],data["labels"],data['mask'],data['ezy_keys'], data['waterlengths'],data['seqlength'],data['maxlength']
profeats,labels,sub_feats,sub_name,ids = data["profeats"],data["labels"],data["subfeats"],data["subname"],data["ids"]
#print(labels,len(labels))

x = data['subname']
folds = list(KFold(n_splits=10, shuffle=True, 
                   random_state=42
                   ).split(th.arange(len(x))))#index = th.arange(len(x)) 
th.save(folds, '')
train_idx, val_idx = folds[0]
print(train_idx,len(train_idx),val_idx,len(val_idx))