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
from final_fusiondataset import*
from Hyd_Km_models import *
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

seed = 88
# PyTorch
torch.manual_seed(seed)          
torch.cuda.manual_seed_all(seed) 
# NumPy
np.random.seed(seed)
# Python
random.seed(seed)
#init
base_lr = 0.001
b_s = 128
init_epoch = 1000
max_epoch = init_epoch
train_losses = []

weights_path = f"path/km_weights_fusion.pth"
scalser_path = f"path/km_scalers_fusion.pkl"
curve_path = f"path/fusion_loss_curve.png"
attention_path = "path/atten_weights.npy"

full_data_path = "path/fulldata_trainV2.pth"

train_loader,dataset,wfeat_scaler,pfeat_scaler,sfeat_scaler,label_scaler = load_full_data_std(full_data_path,batch_size=b_s,is_train=True,scaler=None)


def train(epoch,train_loader,model,device="cuda:0"):
    ww = []
    model.train()
    total_loss = 0.0  
    num_batches = 0      
    num_valbatches = 0
    for batch_idx,(wfeats,fpfeats,fsfeats,labels) in enumerate(train_loader):

        wfeatures = wfeats.to(device)
        pfeatures = fpfeats.to(device)
        sfeatures = fsfeats.to(device)
        labels = labels.to(device)
        
        y_hat,attention_weights = model(wfeatures,pfeatures,sfeatures)
    
        loss = model.huber_loss(y_hat, labels)
    
        l2_reg = model.l2_regularization()

        loss += model.l2_lambda * l2_reg
        
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        

        total_loss += loss.detach().item()    
        num_batches += 1
        ww.append(attention_weights.detach().cpu().numpy())
        
    avg_loss = total_loss / num_batches  
    train_losses.append(avg_loss)
    aw = np.concatenate(ww, axis=0)


    print(f" Epoch {epoch+1},Average Loss: {avg_loss:.4f}",f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    return model,aw 


def train_process(model,train_loader):
    s = time.time()
    scheduler = StepLR(optimizer, step_size=101, gamma=0.5)

    for epoch in range(max_epoch):
        model, aw = train(epoch,train_loader,model)
        scheduler.step()

    e = time.time()
    print(int((e-s) // 60),'m',int((e-s)%60),'s')
    
    
    torch.save(model.state_dict(), weights_path)
    np.save(attention_path, aw)
    joblib.dump({'wfeat': wfeat_scaler,'pfeat': pfeat_scaler,'sfeat': sfeat_scaler,'label': label_scaler}, scalser_path)
    
    return model,epoch



model = Hyd_Km_CM()

optimizer = model.get_optimizer()

train_process(model, train_loader)


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.title('Training Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(curve_path)
plt.close()

