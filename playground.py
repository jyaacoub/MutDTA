# %%
from src.analysis.figures import tbl_dpkd_metrics_n_mut, tbl_stratified_dpkd_metrics
MODEL = lambda i: f"results/model_media/test_set_pred/GVPLM_PDBbind{i}D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE_PLATINUM.csv"
TRAIN_DATA_P = lambda set: f'/cluster/home/t122995uhn/projects/data/PDBbindDataset/nomsa_aflow_gvp_binary/{set}0/cleaned_XY.csv'
NORMALIZE = True


#%%
# add in_binding info to df
def get_in_binding(df, dfr):
    """
    df is the predicted csv with index as <raw_idx>_wt (or *_mt) where raw_idx 
    corresponds to an index in dfr which contains the raw data for platinum including 
    ('mut.in_binding_site')
        - 0: wildtype rows
        - 1: in pocket
        - 2: outside of pocket
    """
    in_pocket = dfr[dfr['mut.in_binding_site'] == 'YES'].index   
    pclass = []
    for code in df.index:
        if '_wt' in code:
            pclass.append(0)
        elif int(code.split('_')[0]) in in_pocket:
            pclass.append(1)
        else:
            pclass.append(2)
            
    df['in_pocket'] = pclass
    return df

# get df_binding info from /cluster/home/t122995uhn/projects/data/PlatinumDataset/raw/platinum_flat_file.csv
conditions = ['(in_pocket == 0) | (in_pocket == 1)', '(in_pocket == 0) | (in_pocket == 2)']
names = ['mutation in pocket', 'mutation NOT in pocket']
#%%
import pandas as pd
dfr = pd.read_csv('/cluster/home/t122995uhn/projects/data/PlatinumDataset/raw/platinum_flat_file.csv', index_col=0)
dfp = pd.read_csv(MODEL(0), index_col=0)

df = get_in_binding(dfp, dfr)
print(df.in_pocket.value_counts())

#%%
tbl_stratified_dpkd_metrics(MODEL, NORMALIZE, n_models=5, df_transform=get_in_binding,
                                       conditions=conditions, names=names, verbose=True, plot=True, dfr=dfr)
# %%
