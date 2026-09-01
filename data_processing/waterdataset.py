import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

excel_path = ""
pt_path = ""
water_path = ""
sub_path = ""
save_path=''
#########################################################################through tables
def create_and_save_dataset(excel_path, sub_dir, pt_dir, water_dir, save_path):
    """Excel&.pt TensorDataset"""
    
    df = pd.read_excel(excel_path,sheet_name="Sheet1")
    sub_features, water_features, pro_features, labels, uni_ids = [], [], [], [], []
    
    for _, row in df.iterrows():
        # 
        sub_feat = torch.load(f'{sub_dir}/{row["full_sub"]}.pt',weights_only=False)
        water_feat = torch.load(f'{water_dir}/{row["full_hyd"]}.pt',weights_only=False)
        pro_feat = torch.load(f'{pt_dir}/{row["full_esm"]}.pt',weights_only=False)

        sub_feat = torch.from_numpy(sub_feat["embedding"])
        sub_features.append(sub_feat)
        water_features.append(water_feat)
        pro_features.append(pro_feat)
    #print(type(water_features),type(water_features[0]),water_features[0])    
        labels.append(row['Kmlog'])
        uni_ids.append(row['IDs'])

    
    # 
    torch.save({
    'wfeats': torch.stack(water_features),
    "pfeats": torch.stack(pro_features),
    "sfeats": torch.stack(sub_features),  # tensor stack
    'labels': torch.tensor(labels),
    'uni_ids': uni_ids,
    }, save_path)
    print(f'saved {save_path}')

#########################################################################
#########################################################################single
# def create_water_dataset(excel_path,water_dir,save_path):   
#     df = pd.read_excel(excel_path)
#     water_features,labels,uni_ids = [], [], []
    
#     for _, row in df.iterrows():
#         
#         water_feature = torch.load(f'{water_dir}/{row["highconfwater"]}.pt')
#         water_features.append(water_feature)
#         labels.append(row['Kmlog'])
#         uni_ids.append(row['uniprot'])
    
#     
#     torch.save({
#     'waterfeats': torch.stack(water_features),
#     'labels': torch.tensor(labels),
#     'uni_ids': uni_ids
#     }, save_path)
#     print(f'saved {save_path}')
#########################################################################
#
create_and_save_dataset(excel_path, sub_path, pt_path, water_path, save_path)
