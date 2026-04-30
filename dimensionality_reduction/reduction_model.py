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


#input_dim = input_dim  
latent_dim = 256  
hidden_dim = 1024  

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, encoding_dim))  
        
        self.dropout = nn.Dropout(0.3)
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim))
    
        self.l2_lambda = 0.01
   
    def forward(self,x):
        x = self.encoder(x)
        x = self.dropout(x)
        y = self.decoder(x)
        return x, y
    
    def loss_fn(self,y,x):
        #loss = ((x - y)**2).sum(-1).sqrt().mean()
        loss = F.mse_loss(y, x)
        return loss
    
    def cosine_similarity_loss(self,y, x):
        similarity = torch.cosine_similarity(y, x)
        loss = 1 - similarity.mean()
        return loss
    
    def get_optimizer(self):
        # self.optimizer = optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        self.optimizer = optim.Adam(self.parameters(), lr=.001, weight_decay = 1e-5)
        return self.optimizer
    
    def l2_regularization(self):
        #
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = param.norm(2)
            else:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg