# %%
import pandas as pd

df = pd.read_csv('/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/train/XY.csv', index_col=0)
dft = pd.read_csv('/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/test/XY.csv', index_col=0)
dfv = pd.read_csv('/cluster/home/t122995uhn/projects/data/DavisKibaDataset/davis/nomsa_binary_original_binary/val/XY.csv', index_col=0)

trainp = df['prot_id'].drop_duplicates()
testp = dft['prot_id'].drop_duplicates()
valp = dfv['prot_id'].drop_duplicates()

overlap_train_test = trainp[trainp.isin(testp)]
overlap_train_val = trainp[trainp.isin(valp)]
overlap_test_val = testp[testp.isin(valp)]

# %%
