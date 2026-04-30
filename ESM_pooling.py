import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob    

# esm_path = ".pt"
# data = torch.load(esm_path,weights_only=False)
# embed_tensor = data['representations'][33]
# #print(embed_tensor)
# features_normalize = np.array([torch.mean(embed_tensor, dim=0).cpu().numpy()])
# #print(features_normalize)

# path
input_folder = "path/wzc_organized_embeddings_output"  
output_folder = "pathr/esm_mean_embeddings_output"  

os.makedirs(output_folder, exist_ok=True)

# all ".pt"
pt_files = glob.glob(os.path.join(input_folder, "*.pt"))

# 
for pt_file in pt_files:
    try:
        # 
        data = torch.load(pt_file, weights_only=False)
        embed_tensor = data['representations'][33]
        
        # 
        features_normalize = torch.mean(embed_tensor, dim=0).cpu()
        
        # 
        base_name = os.path.basename(pt_file)   
        name_without_ext = os.path.splitext(base_name)[0]   
        new_filename = f"{name_without_ext}_feature.pt"   
        output_path = os.path.join(output_folder, new_filename)  
        
        #
        torch.save(features_normalize, output_path)
        print(f"finish: {base_name} -> {new_filename}")
        
    except Exception as e:
        print(f" {pt_file} : {str(e)}")

print("Done")
