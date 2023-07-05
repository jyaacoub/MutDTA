#%%
import pandas as pd
import numpy as np
from src.data_processing import PDBbindProcessor

data = {}
index_file= '../data/v2020-other-PL/index/INDEX_general_PL_name.2020'
line = '6mu1  2018  P29994  INOSITOL 1,4,5-TRISPHOSPHATE RECEPTOR TYPE 1'
#%%
na = []
with open(index_file, 'r') as f:
    for line in f.readlines():
        if line.startswith('#'): continue
        code = line[:4]
        try:
            uniprot = line[11:19].strip()
        except ValueError as e:
            print(f'Error with line: {line}')
            raise e
        if uniprot[0] != '-': # Not provided
            data[code] = uniprot
        else:
            na.append(code)

df = pd.DataFrame.from_dict(data, orient='index', 
                            columns=['uniprot'])
df.index.name = 'PDBCode'

# df.to_csv('./pdb_uniprotID.csv')

# # %%
# with open('./unique_uniprotIDs.txt', 'w') as f:
#     for u in df.uniprot.unique():
#         f.write(f'{u}\n')
# # %%
# with open('./missing_uniprotID_pdbcodes.txt', 'w') as f:
#     for u in na:
#         f.write(f'{u}\n')

#%%
df_seq = pd.read_csv('/home/jyaacoub/projects/data/pytorch_PDBbind/processed/XY.csv', index_col=0)

# %%
out_dir = './'


for code in df_seq.index:
    with open(f'{out_dir}/{code}.fa', 'w') as f:
        seq = df_seq.loc[code].prot_seq
        f.write(f'>{code}\n{seq}')
    break
    
# %%
