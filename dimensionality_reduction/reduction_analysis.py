from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
import seaborn as sns
import umap
from scipy import stats
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import mutual_info_regression

seed = 42
np.random.seed(seed)
raw_path = "path/fulldata(10f).pth"
pro_re_path = "path/AE_prodata_84.pth"
sub_re_path = "path/AE_subdata_84.pth"
file_path = "path/simi-analysis-diag-wp.csv"

data = torch.load(raw_path,weights_only=False)
#data2 = torch.load(pro_re_path,weights_only=False)
#data3 = torch.load(sub_re_path,weights_only=False)

embeddings1 = data['wfeats'].numpy()
embeddings2 = data['p_re_feats'].numpy()
embeddings3 = data['s_re_feats'].numpy()
#print(embeddings1.shape,embeddings2.shape,embeddings3.shape)
#######################################################################################
#1
# df = pd.read_csv(file_path)
# #print(df)
# t, p = stats.ttest_1samp(df["wp_cos"], 0)
# #print(t,p)

# #2
# eb3 = embeddings3.flatten()
# eb2 = embeddings2.flatten()
# pcc, p_value = pearsonr(eb3, eb2)
# r2 = r2_score(eb3, eb2)
# print(pcc,p_value,r2)

#3
cca = CCA(n_components=84)  
A_cca, B_cca = cca.fit_transform(embeddings2, embeddings3)

corrs = [np.corrcoef(A_cca[:, i], B_cca[:, i])[0, 1] for i in range(84)]
print(f"\n (CCA):")
for i, corr in enumerate(corrs):
    print(f"  {i+1}: ρ = {corr:.4f}")
print(f"  first_core: {corrs[0]:.4f}")        

##################################################################################################
# name = data1["sample_names"]
# labels = data1["labels"].numpy().tolist()
# dfname = pd.DataFrame(name,labels)
# dfname.to_csv("/labelsname.csv")
# print(dfname)
##################################################################################################
#5：sklearn
similarities = cosine_similarity(embeddings1, embeddings2)  # (n_samples, n_samples)
diag_similarities = np.diag(similarities)  

n_samples = 50
idx = np.random.choice(8385, n_samples, replace=False)
#idx = np.arange(n_samples)
sampled_matrix = similarities[idx, :][:, idx]
print(diag_similarities)


plt.figure(figsize=(10, 8))
sns.heatmap(sampled_matrix, annot=False, fmt='.2f', cmap='coolwarm', 
            xticklabels=range(50), yticklabels=range(50))
plt.title('Cosine-Similarity')
plt.xlabel('feats(Hydra-84dims)')
plt.ylabel('feats(Substrate-84dims)')
plt.show()

#df1 = pd.DataFrame(diag_similarities)
df2 = pd.DataFrame(similarities)
#df1.to_csv(".../simi-analysis-diag-ps.csv", index=False)
df2.to_csv(".../simi-analysis-ss.csv", index=False)
##################################################################################################
# #2
# tsne_w = TSNE(n_components=2).fit_transform(embeddings1)
# tsne_p = TSNE(n_components=2).fit_transform(embeddings2)


# 
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# scatter1 = ax1.scatter(tsne_p[:,0], tsne_p[:,1], c=y_true, cmap='viridis')
# ax1.set_title('Original')
# scatter2 = ax2.scatter(tsne_w[:,0], tsne_w[:,1], c=y_true, cmap='viridis')
# ax2.set_title('Reduced')
# plt.colorbar(scatter1, ax=[ax1, ax2], label='value')
##################################################################################################
# 
all_embeddings = np.vstack([embeddings1, embeddings2, embeddings3])  # (24000, 84)
labels = np.array([0]*8385 + [1]*8385 + [2]*8385)  # group label

# UMAP
print("start_UMAP...")

# para
reducer = umap.UMAP(
    n_neighbors=30,      
    min_dist=0.3,        
    n_components=3,      
    metric='cosine',     
    random_state=42,     
    verbose=True         
)
embedding_3d = reducer.fit_transform(all_embeddings)
# visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# costume
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  
group_names = ['Group 1', 'Group 2', 'Group 3']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, color in enumerate(colors):
    mask = labels == i
    points = embedding_3d[mask]
    ax.scatter(embedding_3d[mask, 0], embedding_3d[mask, 1], embedding_3d[mask, 2],
               c=color, label=f'Group {i+1}', s=5, alpha=0.6)


    # if len(points) > 4:  # ConvexHull
    #     try:
    #         hull = ConvexHull(points)
    #         
    #         ax.scatter(points[hull.vertices, 0], points[hull.vertices, 1], points[hull.vertices, 2],
    #                    c=color, s=50, marker='o', edgecolor='black', linewidth=1)
            
    #         
    #         for simplex in hull.simplices:
    #             simplex = np.append(simplex, simplex[0])
    #             ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2],
    #                     c=color, alpha=0.3, linewidth=0.5)
    #     except:
    #         pass  



    #     # 
    # if len(points) > 3:
    #     # 
    #     gm = GaussianMixture(n_components=1, covariance_type='full')
    #     gm.fit(points)
        
    #     # 
    #     center = gm.means_[0]
    #     cov = gm.covariances_[0]
        
    #     # 
    #     eigenvals, eigenvecs = np.linalg.eigh(cov)
        
    #     ax.scatter(center[0], center[1], center[2], 
    #                c=color, s=200, marker='+', linewidths=3, 
    #                alpha=0.8, label=f'Group {i+1} Center')


ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('UMAP 3D', fontsize=16)
ax.legend()
plt.show()