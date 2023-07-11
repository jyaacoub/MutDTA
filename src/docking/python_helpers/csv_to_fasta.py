#%% imp
import pandas as pd
from tqdm import tqdm
df_seq = pd.read_csv('/cluster/home/t122995uhn/projects/data/XY.csv', index_col=0)

print(df_seq)
# %%
out_dir = '/cluster/projects/kumargroup/jean/sequences'


for code in tqdm(df_seq.index, total=len(df_seq)):
    with open(f'{out_dir}/{code}.fa', 'w') as f:
        seq = df_seq.loc[code].prot_seq
        f.write(f'>{code}\n{seq}')
