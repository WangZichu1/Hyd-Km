import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
import random
import math

seed = 42
#
random.seed(seed)
full_data_path = ''
train_save_path = ""
test_save_path = ""
# tensor
data = torch.load(full_data_path,weights_only=False)
total_samples = data["wfeats"].shape[0]
print(f"total: {total_samples}")
test_size=math.ceil(0.1*total_samples)

test_indices = sorted(random.sample(range(total_samples), test_size))
# training & test set
test_set = set(test_indices)
train_indices = [i for i in range(total_samples) if i not in test_set]

uni_ids = np.array(data["uni_ids"])

# splitter = StratifiedShuffleSplit(
#     n_splits=1,      
#     test_size=0.1,   
#     random_state=42  
# )

#for train_idx, test_idx in splitter.split((data["wfeats"],data["labels"])):
train_wfeats = data["wfeats"][train_indices]
train_fpfeats = data["pfeats"][train_indices]
train_fsfeats = data["sfeats"][train_indices]
train_labels = data["labels"][train_indices]
train_ids = np.array(data["uni_ids"])[train_indices]
#train_names = [data["sample_names"][i] for i in train_indices]    

test_wfeats = data["wfeats"][test_indices]
test_fpfeats = data["pfeats"][test_indices]
test_fsfeats = data["sfeats"][test_indices]
test_labels = data["labels"][test_indices]
test_ids = np.array(data["uni_ids"])[test_indices]
#test_names = [data["sample_names"][i] for i in test_indices] 

train_data = {
    "train_wfeats":train_wfeats,
    "train_fpfeats":train_fpfeats,
    "train_fsfeats":train_fsfeats,
    "train_labels":train_labels,
    "train_ids":train_ids
}
test_data = {
    "test_wfeats":test_wfeats,
    "test_fpfeats":test_fpfeats,
    "test_fsfeats":test_fsfeats,
    "test_labels":test_labels,
    "test_ids":test_ids
}


torch.save(train_data, train_save_path)
torch.save(test_data, test_save_path)

print(f"testnum: {test_size}")
print(f"trainnum: {len(train_indices)}")
print(f"test_range: {min(test_indices)} ~ {max(test_indices)}")
print(f"saved: {test_save_path}")
print(f"saved: {train_save_path}")
