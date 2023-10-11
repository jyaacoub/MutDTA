#%%
import pandas as pd
import os
from glob import glob

data = 'davis'
fp = f'/cluster/home/t122995uhn/projects/data/DavisKibaDataset/{data}/nomsa_binary/full/XY.csv'
done_dir = f'/cluster/home/t122995uhn/projects/colabfold/{data}_af2_out/highQ/*.done.txt'

df = pd.read_csv(fp)

#%%
# replace '(' with '_' in prot_id col
df['pid_fix'] = df['prot_id'].str.replace(r'[()]', '_', regex=True)
unique_pids = df['pid_fix'].unique()
print(f'{len(unique_pids)} unique proteins')

def getID(fn):
    return os.path.splitext(os.path.basename(fn))[0].split('.done')[0]

done_IDs =  {getID(filename) for filename in glob(done_dir)}

remaining = [id for id in unique_pids if id not in done_IDs]

print(len(remaining), 'protiens remaining')

done_rows = df.iloc[df[~df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows = df.iloc[df[df['pid_fix'].isin(remaining)][['pid_fix']].drop_duplicates().index]
remaining_rows['len'] = remaining_rows['prot_seq'].str.len()

max_prot_len_done = done_rows['prot_seq'].str.len().max()
max_prot_len = remaining_rows['len'].max()
print(f'{max_prot_len} is the max protein length for remaining structures.\n{max_prot_len_done} for done structures')
remaining_rows[['pid_fix', 'len']]

#%% get pids of those above 2149
files_to_remove = remaining_rows[remaining_rows['len'] > 2000]['prot_id']

for i in range(1,4):
    print('\nPART', i)
    p3p = f'/cluster/home/t122995uhn/projects/colabfold/{data}_a3m/part{i}'

    for f in files_to_remove:
        try:
            os.remove(os.path.join(p3p, f'{f}.a3m'))
            print(f'{f} REMOVED')
        except FileNotFoundError:
            pass
            # print(f'{f} doesnt exist')