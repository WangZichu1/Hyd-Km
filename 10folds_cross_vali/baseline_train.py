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
from torch.optim import optimizer
from baseline_dataset import loaddata_std
from Hyd_Km_models import *
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

seed = 88
torch.manual_seed(seed)          
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)
random.seed(seed)


for i in range(10):

    fold_id = i
    base_lr = 0.001
    b_s = 128
    init_epoch = 1000
    max_epoch = init_epoch
    train_losses = []
    val_losses = []

    weights_path = f"path/km_weights_ES{i}.pth"
    scalser_path = f"path/km_scalers_ES{i}.pkl"
    curve_path = f"path/km_weights_ES{i}.png"

    val_prepath = f'path/km_predictions_ES{i}.npy'
    val_labelpath = f'path/km_val_labels_ES{i}.npy'

    idx_path = "path/index_fulldata_final_train_10folds.pth"
    full_data_path = "path/fulldata_trainV2.pth"

    train_loader,valid_loader,train_dataset,valid_dataset,feat_scaler,label_scaler = loaddata_std(full_data_path,idx_path,fold_id,batch_size=b_s,is_train=True,scaler=None)


    def train(epoch,train_loader,valid_loader,model,device="cuda:0"):

        model.train()
        total_loss = 0.0  
        num_batches = 0      
        num_valbatches = 0
        for batch_idx,(feats,labels) in enumerate(train_loader):#enumerate

            features = feats.to(device)
            labels = labels.to(device)
            
            y_hat = model(features)
        
            loss = model.huber_loss(y_hat, labels)
        
            l2_reg = model.l2_regularization()

            loss += model.l2_lambda * l2_reg
            
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            total_loss += loss.detach().item()    
            num_batches += 1
        avg_loss = total_loss / num_batches  
        train_losses.append(avg_loss)


        model.eval()
        valid_loss = 0.0
        all_val_pre = []  
        all_val_labels = []  
        with torch.no_grad():

            for batch_idx,(feats,labels) in enumerate(valid_loader):

                val_wfeatures = feats.to(device)
                val_labels = labels.to(device)
    
                val_y_hat = model(val_wfeatures)
                val_pre = val_y_hat.cpu().numpy()

                all_val_pre.append(val_pre)
                all_val_labels.append(val_labels.cpu().numpy())
                
                loss = model.huber_loss(val_y_hat, val_labels)
                valid_loss += loss.detach().item()
                num_valbatches +=1
            
            all_val_pre = np.concatenate(all_val_pre, axis=0)
            all_val_labels = np.concatenate(all_val_labels, axis=0)
            
            avg_val_loss = valid_loss/num_valbatches
            val_losses.append(avg_val_loss)       
        
        print(f" Epoch {epoch+1},Average Loss: {avg_loss:.4f}",f"Aveval Loss:{avg_val_loss:.4f},Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        np.save(val_prepath, all_val_pre)
        np.save(val_labelpath, all_val_labels)
        return model,all_val_pre, all_val_labels


    def train_process(model,train_loader,valid_loader):
        s = time.time()
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        for epoch in range(max_epoch):
            train(epoch,train_loader,valid_loader,model)
            scheduler.step()
        e = time.time()
        print(int((e-s) // 60),'m',int((e-s)%60),'s')
        

        torch.save(model.state_dict(), weights_path)  
        joblib.dump({'feat': feat_scaler,'label': label_scaler}, scalser_path)
        
        return model,epoch

    model = Catapro_Km_model()

    optimizer = model.get_optimizer()

    train_process(model, train_loader,valid_loader)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(curve_path)
    plt.close()

