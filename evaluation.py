import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest
from sklearn.metrics import r2_score
#for i in range(10):
# 
# all_val_pre = np.load(f"path/ofusion_predicted.npy")
# all_val_labels = np.load(f"path/ofusion_label.npy")

# 
# print("all_val_pre shape:", all_val_pre.shape)
# print("all_val_labels shape:", all_val_labels.shape)
# 
# # predicted_labels = (all_val_pre > 0.5).astype(int)
# 
# # predicted_labels = np.argmax(all_val_pre, axis=1)
# 
# # predicted_values = all_val_pre

# 
# all_val_pre_flat = all_val_pre.flatten()
# all_val_labels_flat = all_val_labels.flatten()
# 
# # df1 = pd.DataFrame({'True_Label': all_val_labels,})
# # df2 = pd.DataFrame({'Predicted_Value': all_val_pre})
# 
# df = pd.DataFrame({
#     'True_Label': all_val_labels_flat,
#     'Predicted_Value': all_val_pre_flat
# })

# 
# print(df.head())
# 
# df.to_csv(f'path/results_ofusion.csv', index=False)
# #############################################################################################################

#############################################################################################################
# CSV 
df = pd.read_csv("path/results_ofusion.csv")

# 
predicted = df["Predicted_Value"]
true = df["True_Label"]

# PCC
pcc, _ = pearsonr(predicted, true)
print(f"Pearson Correlation Coefficient (PCC): {pcc}")
# Scc
scc, _ = spearmanr(predicted, true)
print(f"Spearman Correlation Coefficient (Scc): {scc}")
# RMSE
rmse = np.sqrt(np.mean((predicted - true) ** 2))
print(f"Root Mean Square Error (RMSE): {rmse}")
#mae 
mae = np.mean(np.abs(predicted - true))
print(f"Mean Absolute Error (MAE): {mae}")
# R²
r2 = r2_score(true, predicted)
print(f"R-squared (R²): {r2}")
#mape
mape = np.mean(np.abs((true - predicted) / true)) * 100
print(mape)
#smape
smape = np.mean(np.abs(true - predicted) / (np.abs(true) + np.abs(predicted))) * 100
print(smape)

valuestates = df["Predicted_Value"].describe()
skewness = df["Predicted_Value"].skew() 
kurtosis = df["Predicted_Value"].kurt()
print(valuestates,f'Skewness: {skewness}',f'Kurtosis: {kurtosis}')

#
plt.figure(figsize=(10, 6))
sns.histplot(df["Predicted_Value"], kde=True, bins=20)
plt.title('Distribution of Predicted Value')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
#
plt.figure(figsize=(8, 6))
sns.boxplot(x=df["Predicted_Value"])
plt.title('Boxplot of Value')
plt.xlabel('Value')
plt.show()