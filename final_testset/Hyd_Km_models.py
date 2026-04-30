import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
import math
from torch.func import functional_call, vmap, grad
from torch.autograd import Variable
from functools import partial

class Single_modality_model(nn.Module):
    def __init__(self, rate=0.3, device="cuda:0"):#rate =0.3 or 0.5
        super(Single_modality_model,self).__init__()
       #input_dim = 1280\768\84
        self.feats_norm = nn.LayerNorm(1280).to(device)
        self.decoder = nn.Sequential(nn.Linear(1280, 256), 
                                     nn.LayerNorm(256), 
                                     nn.Dropout(p=rate), 
                                     nn.LeakyReLU(),
                                #nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(128, 32), nn.LayerNorm(32), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p=rate),nn.LeakyReLU()
                                ).to(device)
        
        self.out = nn.Sequential(nn.Linear(16, 1)).to(device)#
        self.l2_lambda = 0.001

    def forward(self, reduced_feats):
        reduced_feats = self.feats_norm(reduced_feats)

        feats = self.decoder(reduced_feats)

        out = self.out(feats)

        return out 
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg

  
class Catapro_Km_model(nn.Module):
    def __init__(self, rate=0.3, device="cuda:0"):
        super(Catapro_Km_model,self).__init__()
        
        self.feats_norm = nn.BatchNorm1d(2048).to(device)
 
        self.decoder = nn.Sequential(nn.Linear(2048, 256), 
                                     nn.BatchNorm1d(256), 
                                     nn.Dropout(p=rate), 
                                     nn.ReLU(),
                                ).to(device)
        
        self.out = nn.Sequential(nn.Linear(256, 1)).to(device)#
        
        self.l2_lambda = 0.001

    def forward(self, reduced_feats):
        reduced_feats = self.feats_norm(reduced_feats)
        feats = self.decoder(reduced_feats) 

        out = self.out(feats)

        return out 
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg

class Concat_Tri_Reduced(nn.Module):
    def __init__(self, rate=0.3, device="cuda:0"):
        super(Concat_Tri_Reduced,self).__init__()
#########################################################set (252) to (168) and get Baseline        
        self.feats_norm = nn.LayerNorm(252).to(device)
    
        self.decoder = nn.Sequential(nn.Linear(252, 256), 
                                     nn.LayerNorm(256), 
                                     nn.Dropout(p=rate), 
                                     nn.LeakyReLU(),
                                #nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(128, 32), nn.LayerNorm(32), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p=rate),nn.LeakyReLU()
                                ).to(device)
        
        self.out = nn.Sequential(nn.Linear(16, 1)).to(device)#
        # L2
        self.l2_lambda = 0.001

    def forward(self, reduced_feats):
        reduced_feats = self.feats_norm(reduced_feats)

        feats = self.decoder(reduced_feats) 

        out = self.out(feats)

        return out #feats
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg
    
class Concat_Tri(nn.Module):
    def __init__(self, rate=0.3, device="cuda:0"):
        super(Concat_Tri,self).__init__()
        
        self.feats_norm = nn.BatchNorm1d(2132).to(device)
        
        self.decoder = nn.Sequential(nn.Linear(2132, 256), 
                                     nn.BatchNorm1d(256), 
                                     nn.Dropout(p=rate), 
                                     nn.ReLU(),
                                ).to(device)
        
        self.out = nn.Sequential(nn.Linear(256, 1)).to(device)#
        #
        self.l2_lambda = 0.001

    def forward(self, reduced_feats):
        reduced_feats = self.feats_norm(reduced_feats)

        feats = self.decoder(reduced_feats) 

        out = self.out(feats)

        return out #feats
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg

class Hyd_Km_DR(nn.Module):
    def __init__(self,fused_dim=84,rate=0.3,device = "cuda:0"):
        super(Hyd_Km_DR, self).__init__()
        self.proj_w =nn.Sequential(nn.Linear(84, fused_dim),
                                   nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_p = nn.Sequential(nn.Linear(84, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_s = nn.Sequential(nn.Linear(84, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)

        self.attention_fc = nn.Linear(fused_dim * 3, 3).to(device)
        
        self.final_out = nn.Sequential(nn.Linear(fused_dim * 3, 256), 
                                     nn.LayerNorm(256), 
                                     nn.Dropout(p=rate), 
                                     nn.LeakyReLU(),
                                #nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(256, 128), nn.LayerNorm(128), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(128, 32), nn.LayerNorm(32), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(16, 1)
                                ).to(device) 

        self.l2_lambda = 0.001

    def forward(self, feat_w, feat_p, feat_s):
        #1.
        w_proj = self.proj_w(feat_w)
        p_proj = self.proj_p(feat_p)
        s_proj = self.proj_s(feat_s)        
        # 2. 
        combined = torch.cat([w_proj, p_proj, s_proj], dim=-1)        
        # 3. 
        att_weights = F.softmax(self.attention_fc(combined), dim=-1) # [batch_size, 3]        
        # 4. fusion
        # broadcast
        w_w, w_p, w_s = att_weights[:, 0:1], att_weights[:, 1:2], att_weights[:, 2:3]
        #fused_feat = w_w * w_proj + w_p * p_proj + w_s * s_proj
        fused_feat = torch.cat([w_w * w_proj, w_p * p_proj, w_s * s_proj], dim=-1)       
        # 5. 
        output = self.final_out(fused_feat)
        return output #att_weights 
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):#delta
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):

        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg


class Hyd_Km_OriandWO(nn.Module):
    def __init__(self,fused_dim=84,rate=0.3,device = "cuda:0"):
        super(Hyd_Km_OriandWO, self).__init__()
        # shared projection space
        self.proj_w =nn.Sequential(nn.Linear(84, fused_dim),
                                   nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_p = nn.Sequential(nn.Linear(1280, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_s = nn.Sequential(nn.Linear(768, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.att_q_w = nn.Linear(fused_dim, 1).to(device)  # [B, D] -> [B, 1]
        self.att_q_p = nn.Linear(fused_dim, 1).to(device) # [B, D] -> [B, 1]
        self.att_q_s = nn.Linear(fused_dim, 1).to(device)  # [B, D] -> [B, 1
#####################################################################################        
        #self.attention_fc = nn.Linear(fused_dim * 3, 3).to(device) #Version 1 
#####################################################################################        
        self.final_out = nn.Sequential(nn.Linear(252, 256), 
                                     nn.LayerNorm(256), 
                                     nn.Dropout(p=rate), 
                                     nn.LeakyReLU(),
                                #nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(256, 128), nn.LayerNorm(128), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(128, 32), nn.LayerNorm(32), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(256, 1)
                                ).to(device) # final

                # L2
        self.l2_lambda = 0.001

    def forward(self, feat_w, feat_p, feat_s):
         
        w_proj = self.proj_w(feat_w)
        p_proj = self.proj_p(feat_p)
        s_proj = self.proj_s(feat_s)
         
        scores = torch.stack([
        self.att_q_w(w_proj).squeeze(-1),  
        self.att_q_p(p_proj).squeeze(-1),
        self.att_q_s(s_proj).squeeze(-1)
        ], dim=-1)  
        att_weights = F.softmax(scores, dim=-1)
################################################################################################               
        # combined = torch.cat([w_proj, p_proj, s_proj], dim=-1)       
        # att_weights = F.softmax(self.attention_fc(combined), dim=-1) # [batch_size, 3]   #Version 1      
################################################################################################
        w_w, w_p, w_s = att_weights[:, 0:1], att_weights[:, 1:2], att_weights[:, 2:3]
        #fused_feat = w_w * w_proj + w_p * p_proj + w_s * s_proj
        fused_feat = torch.cat([w_w * w_proj, w_p * p_proj, w_s * s_proj], dim=-1)     #  Hyd_Km_Ori  
        #fused_feat = torch.cat([w_w * feat_w, w_p * feat_p, w_s * feat_s], dim=-1)    #  Hyd_Km_WO
        output = self.final_out(fused_feat)
        return output#att_weights 
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        #
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg
    
class Hyd_Km_CM(nn.Module):
    def __init__(self,fused_dim=128,rate=0.3,device = "cuda:0"):
        super(Hyd_Km_CM, self).__init__()
        # shared projection space
        self.proj_w =nn.Sequential(nn.Linear(84, fused_dim),
                                   nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_p = nn.Sequential(nn.Linear(1280, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.proj_s = nn.Sequential(nn.Linear(768, fused_dim),
                                    nn.LayerNorm(fused_dim),
                                   nn.Dropout(p=rate),
                                   nn.LeakyReLU(),).to(device)
        self.att_q_w = nn.Linear(fused_dim*2, fused_dim).to(device)  
        self.att_q_p = nn.Linear(fused_dim*2, fused_dim).to(device) 
        self.att_q_s = nn.Linear(fused_dim*2, fused_dim).to(device)  
        
        self.final_out = nn.Sequential(#nn.LayerNorm(768),
                                     nn.Linear(384, 256), 
                                     nn.LayerNorm(256), 
                                     nn.Dropout(p=rate), 
                                     nn.LeakyReLU(),
                                #nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(256, 128), nn.LayerNorm(128), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(128, 32), nn.LayerNorm(32), nn.Dropout(p=rate), nn.LeakyReLU(),
                                #nn.Linear(32, 16), nn.LayerNorm(16), nn.Dropout(p=rate), nn.LeakyReLU(),
                                nn.Linear(256, 1)
                                ).to(device) 

                # L2
        self.l2_lambda = 0.001

    def forward(self, feat_w, feat_p, feat_s):
         
        w_proj = self.proj_w(feat_w)
        p_proj = self.proj_p(feat_p)
        s_proj = self.proj_s(feat_s)
        
        ps = torch.cat([p_proj,s_proj],dim=-1)
        ws = torch.cat([w_proj,s_proj],dim=-1)
        wp = torch.cat([w_proj,p_proj],dim=-1)       
        
        score_w = F.softmax(self.att_q_w(ps),dim=-1)
        score_p = F.softmax(self.att_q_p(ws),dim=-1)
        score_s = F.softmax(self.att_q_s(wp),dim=-1)

        fused_feat = torch.cat([score_w * w_proj, score_p * p_proj, score_s * s_proj], dim=-1)      
        #ori_feat= torch.cat([w_proj,p_proj,s_proj], dim=-1)

        output = self.final_out(fused_feat)
        return output#score_w 
    
    def get_optimizer(self, #learning_rate, momentum, weight_decay
                      ):
        #self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001, weight_decay = 1e-5)
        return self.optimizer
    
    def huber_loss(self, out, y, delta=1.5):#
        error =  out - y
        condition = torch.abs(error) < delta
        loss = torch.where(
            condition,
            0.5 * error**2,
            delta * (torch.abs(error) - 0.5 * delta)
        )
        return torch.mean(loss)
    
    def loss_fn(self,y,x):
        loss = ((x - y)**2).sum(-1).sqrt().mean()
        #loss = ((y - x)**2).mean().sqrt()
        #loss = F.mse_loss(y, x, reduction='sum')
        return loss

    def l2_regularization(self):
        
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg



