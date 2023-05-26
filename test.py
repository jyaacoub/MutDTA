# %%
from utils import download_PDBs, get_df
import pandas as pd
import json

X_path = './data/PDBbind/kd_ki/X.csv'
X = pd.read_csv(X_path)

codes = X['PDBCode'].values

#%%
downloaded_codes = download_PDBs(codes[:10], './data/structures')

with open('./data/downloaded_codes.json', 'w') as f:
    json.dump(downloaded_codes, f)
    
    


