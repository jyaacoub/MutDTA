#%% testing download of sequences from FASTA files
import requests as r
from data_processing.pdbbind import get_prot_seq, save_prot_seq, excel_to_csv, prep_save_data
import pandas as pd
import os, re

# %% TEST for Ki vs Kd:
csv_path='data/PDBbind/raw/P-L_refined_set_all.csv'
df_raw = pd.read_csv(csv_path)
df = df_raw[lambda x: x['protID'].fillna('').str.split().apply(lambda x: len(x)==1)]
df_a = df.affinity

conv = {
    'mM': 1000,
    'uM': 1,
    'nM': 1e-3,
    'pM': 1e-6,
    'fM': 1e-9,
}
const = {
    'Ki':[],
    'Kd':[]
}

for a in df_a:
    k, v = re.split(r'=|<=|>=', a)
    v = float(v[:-2]) * conv[v[-2:]]
    if 'Ki' in a:
        const['Ki'].append(v)
    elif 'Kd' in a:
        const['Kd'].append(v)
    else:
        print(a)

# removing outliers (Ki or Kd > 1e-3)
for k in const:
    const[k] = [x for x in const[k] if x < 1e-3]

import matplotlib.pyplot as plt

plt.hist(const['Ki'], bins=10, alpha=0.8, color='r')
plt.hist(const['Kd'], bins=10, alpha=0.8, color='g')

plt.legend(['Ki', 'Kd'])
print('Ki:',len(const['Ki']), 'Kd:', len(const['Kd']))


from scipy.stats import ranksums
listl = const['Ki']
list2 = const['Kd']
u_stat, p_val = ranksums(listl, list2)
print('p-value:', p_val)
print('u-stat:', u_stat)
