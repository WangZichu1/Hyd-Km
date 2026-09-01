import pandas as pd
import numpy as np

df = pd.read_excel('')

# km log1p
df['Km(M)'] = np.log1p(df['Km(M)'])
#df['Km(M)'] = np.log1p(df['Km(M)'] / np.percentile(df['Km(M)'], 95)) #or

df.to_excel('C:/Users/wangz/Desktop/11-24-water/seq17-601-km-final-water.xlsx', index=False)