#%%
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

df = pd.read_csv("../data/TCGA_BRCA_Mutations.csv")
# %%
df[['Gene', 'SWISSPROT']].head()
# %%
df2 = pd.read_csv("../data/PlatinumDataset/nomsa_binary_original_binary/full/cleaned_XY.csv", index_col=0)
# %%
df2['pdb_id'] = df2.prot_id.str.split("_").str[0]

# %%
from tqdm import tqdm
import requests
from src.utils.pdb import _pdb2uniprot, pdb2uniprot

# uniprots = []
# pdb_ids = df2.pdb_id.unique()
# with ThreadPoolExecutor(max_workers=20) as executor:  # You can adjust the number of workers based on your environment
#     future_to_pid = {executor.submit(_pdb2uniprot, pid): pid for pid in pdb_ids}
#     for future in tqdm(as_completed(future_to_pid), total=len(pdb_ids)):
#         result = future.result()
#         if result:
#             uniprots.append(result[0])

uniprots = pdb2uniprot(df2.pdb_id.unique())
# %%

