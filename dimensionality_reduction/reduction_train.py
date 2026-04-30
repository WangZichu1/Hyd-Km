import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import random
import copy
import torch.optim as optim
import numpy as np
import math
import joblib
import os
from torch.optim import optimizer
from reduction_dataset import loaddata_std
from reduction_model import *
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_all_seeds(42)

#
b_s = 128
init_epoch = 1000
max_epoch = init_epoch
train_losses = []
val_losses = []

reduce_weights_path = "path/AEweights_sub_84.pth"
scalser_path = "path/AEscaler_sub_84.pkl"

full_data_path = 'path/full_data.pth'
pth_path = "path/esm_subdataset.pth"

train_loader,valid_loader,train_dataset, valid_dataset,feat_scaler, label_scaler = loaddata_std(pth_path,full_data_path,batch_size=b_s,is_train=True, scaler=None)

input_dim = 768  
#latent_dim = 64  
encoding_dim = 84
#hidden_dim = 1024

model = Autoencoder(input_dim, encoding_dim).to(device)
#
optimizer = model.get_optimizer()

#
def train(epoch,train_loader,valid_loader,model,device="cuda:0"):
    #TotalBCE = 0
    #TotalKLloss = 0    
    model.train()
    total_loss = 0.0  
    num_batches = 0   
    num_valbatches = 0

    for batch_idx,(feats,labels) in enumerate(train_loader):
        
        features = feats.to(device)
        labels = labels.to(device)

        #x_hat,mu,logvar = model(features)
        x_hat = model(features)[1]

        loss = model.loss_fn(x_hat,features)
        #loss ,kl= model.loss_function(x_hat, features, mu, logvar,beta=1.0)
        
        # L2
        l2_reg = model.l2_regularization()

        loss += model.l2_lambda * l2_reg

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        total_loss += loss.detach().item()
        #TotalBCE +=  bce.detach().item()
        #TotalKLloss += KLloss.detach().item()
        num_batches += 1
    avg_loss = total_loss / num_batches 
    #avg_bce = TotalBCE / num_batches
    #avg_Klloss = TotalKLloss / num_batches
    train_losses.append(avg_loss)


    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch_idx,(feats,labels) in enumerate(valid_loader):
            val_features = feats.to(device)
            val_labels = labels.to(device)
            #val_x_hat,val_mu,val_logvar = model(val_features)
            val_x_hat = model(val_features)[1]
            #val_loss,val_kl = model.loss_function(val_x_hat, val_features, val_mu,val_logvar,beta=1.0)
            val_loss = model.loss_fn(val_x_hat,val_features)
            valid_loss += val_loss.detach().item()
            num_valbatches +=1
        avg_val_loss = valid_loss/num_valbatches
        val_losses.append(avg_val_loss)           

    print(f" Epoch {epoch+1},Average Loss: {avg_loss:.4f}",
          f"Aveval Loss:{avg_val_loss:.4f}",
        #f"KL:{kl.detach().item():.4f}",
        #f"valKL:{val_kl.detach().item():.4f}",
        f"Current Learning Rate:{optimizer.param_groups[0]['lr']}")

#
def train_process(model,train_loader,valid_loader):
    s = time.time()
    scheduler = StepLR(optimizer, step_size=101, gamma=0.5)
    for epoch in range(max_epoch):
        train(epoch,train_loader,valid_loader,model)
        scheduler.step()
    e = time.time()
    print(int((e-s) // 60),'m',int((e-s)%60),'s')
    
    # save
    torch.save(model.state_dict(), reduce_weights_path)
    joblib.dump({'feat': feat_scaler, #'label': label_scaler
                 }, 
                scalser_path)
    
    return model,epoch


train_process(model, train_loader,valid_loader)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()