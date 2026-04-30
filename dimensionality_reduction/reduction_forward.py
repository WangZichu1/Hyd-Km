import torch
import random,os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from reduction_model import *
from reduction_dataset import *

# GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
#init
b_s = 128
encoding_dim = 84
input_dim = 1280
#latent_dim = 400  
#hidden_dim = 1024 
full_data_path = "path/full_data.pth"
save_path = "path/AE_prodata_84.pth"
#######################################################################################################
# def loaddata(pth_path):
#     embeddings = torch.load(pth_path,weights_only=False)    
#     #
#     if isinstance(embeddings, dict):
#         # dict: {'features': ..., 'labels': ...}
#         features = embeddings['feats']
#         labels = embeddings['labels']
#     elif isinstance(embeddings, tuple) and len(embeddings) == 2:
#         # tuple: (features, labels)
#         features, labels = embeddings
#     else:
#         raise ValueError
        
#         # numpy
#     if isinstance(features, torch.Tensor):
#         features = features.numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.numpy()
#     print(f"feature_shape: {features.shape}, label_shape: {labels.shape}")
#     return features, labels
#######################################################################################################
def get_aeforward_data(full_data_path, is_train=False, scaler = None, device ="cuda:0" ):
    data = torch.load(full_data_path,weights_only=False)
    
    wfeats = data["wfeats"]
    pfeats = data["pfeats"]
    sfeats = data["sub_feats"]
    labels = data["labels"]
    subname = data["sub_name"]
    ids = data["ids"]
    #labels = torch.log10(labels)
    #cplx_feats = th.cat([wfeats, sfeats], axis=1)

    pfeatsnp = pfeats.numpy()
    feats_scaled = scaler['feat'].transform(pfeatsnp)
    pfeats_scaled = torch.from_numpy(feats_scaled).float()
    pfeats_scaled = pfeats_scaled.to(device)
    #labelsnp = labels.numpy()
    #labelsnp_scaled = scaler['label'].transform(labelsnp)
    #labels_scaled = torch.from_numpy(labelsnp_scaled).float().view(-1, 1)
  
    dataset = TensorDataset(pfeats_scaled, labels)
    #num_total_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=len(dataset),shuffle=False)

    return data_loader

#######################################################################################################
#save results
def save_results(re_features, labels, save_path):
    
    # dict
    reduced_data = {
        'resfeats': torch.tensor(re_features, dtype=torch.float32),
        
        'labels': torch.as_tensor(labels, dtype=torch.float32)
    }
    
    # save
    torch.save(reduced_data, save_path)
    print(f"save as: {save_path}")
#######################################################################################################
#test
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #scaler
    scaler = joblib.load("path/AEscaler_pro_84.pkl")
    # data
    data_loader = get_aeforward_data(full_data_path, is_train=False, scaler = scaler, device=device)
    #model
    model = Autoencoder(input_dim,encoding_dim).to(device)
    model.load_state_dict(th.load("path/AEweights_pro_84.pth", map_location=device))
    
    #forward
    model.eval()
    with torch.no_grad():
        for step, (pfeats_scaled, labels) in enumerate(data_loader):
            re_features, _ = model(pfeats_scaled)  # on GPU
            re_features = re_features.cpu().numpy()
            labels = labels.cpu()
    print(f"shape: {re_features.shape}")
    save_results(re_features,labels, save_path)
    return re_features    
#####################################################################
#####################################################################
if __name__ == "__main__":
    def set_all_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    set_all_seeds(42)
    main()