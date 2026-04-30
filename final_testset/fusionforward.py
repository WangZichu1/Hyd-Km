import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import random
import os
import copy
import torch.optim as optim
import numpy as np
import math
import joblib
from torch.optim import optimizer
from final_fusiondataset import *
from Hyd_Km_models import *
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# （GPU or CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#
print(f"Using device: {device}")

full_data_path = "path/fulldata_testV2.pth"
model_path = "path/km_weights_fusion.pth"
scaler_path = "path/km_scalers_fusion.pkl"
val_prepath = "path/fusion_predicted.npy"
val_labelpath = "path/fusion_label.npy"
#init
b_s = 839#testsets size

scaler = joblib.load(scaler_path)

model = Hyd_Km_CM()
model.load_state_dict(th.load(model_path, map_location=device))

testloader,dataset = load_full_data_std(full_data_path,batch_size=b_s,is_train=False,scaler=scaler)

def inference(model, dataloader, device="cuda:0"):
    model.eval()
    all_val_pre = []  
    all_val_labels = [] 
    with torch.no_grad():
        for batch_idx,(wfeats,fpfeats,fsfeats,labels) in enumerate(dataloader):
            val_wfeatures = wfeats.to(device)
            val_pfeatures = fpfeats.to(device)
            val_sfeatures = fsfeats.to(device)
            val_labels = labels.to(device)

            val_y_hat,fnattention_weights = model(val_wfeatures,val_pfeatures,val_sfeatures)
            val_pre = val_y_hat.cpu().numpy()

            all_val_pre.append(val_pre)
            all_val_labels.append(val_labels.cpu().numpy())

            np.save(val_prepath, all_val_pre)
            np.save(val_labelpath, all_val_labels)
    return all_val_pre,all_val_labels
            
inference(model, testloader, device="cuda:0")
                