#%%
import seaborn as sns
import pandas as pd
import json
import matplotlib.pyplot as plt
from src.data_analysis.stratify_protein import check_davis_names, kinbase_to_df
from src.utils import config as cfg

df = kinbase_to_df()

# %%
# adding missing protein names/ aliases
# mapping missing proteins to correct ones from kinbase
# or to proteins with the same main family
missing_prots = {
    # TGF superfamily
    'ACVR1':  'ALK2',       # https://www.ncbi.nlm.nih.gov/gene/90
    'ACVR1B': 'ALK4',       # https://www.ncbi.nlm.nih.gov/gene/91
    'ACVR2A': 'ACTR2',      # https://www.ncbi.nlm.nih.gov/gene/92
    'ACVR2B': 'ACTR2B',     # https://www.ncbi.nlm.nih.gov/gene/93
    'ACVRL1': 'ALK1',       # https://www.ncbi.nlm.nih.gov/gene/94
    
    'ANKK1': 'sgk288',      # https://www.ncbi.nlm.nih.gov/gene/255239
    
    'ARK5': 'NuaK1',        # https://www.ncbi.nlm.nih.gov/gene/9891
    
    'ASK1': 'MAP3K5',       # https://en.wikipedia.org/wiki/ASK1
    'ASK2': 'MAP3K6',       # https://www.ncbi.nlm.nih.gov/gene/9064
    
    'AURKA': 'AurA',        # https://www.ncbi.nlm.nih.gov/gene/6790
    'AURKB': 'AurB',        # https://www.ncbi.nlm.nih.gov/gene/9212
    'AURKC': 'AurC',        # https://www.ncbi.nlm.nih.gov/gene/6795
    
    'CDC2L1':   'CDK11',    # https://en.wikipedia.org/wiki/CDC2L1
    'CDC2L2':   'CDK11',    # https://www.ncbi.nlm.nih.gov/gene/493708
    'CDC2L5':   'CHED',     # https://www.ncbi.nlm.nih.gov/gene/8621
    
    'CHEK1':    'CHK1',     # https://www.ncbi.nlm.nih.gov/gene/1111
    'CHEK2':    'CHK2',     # https://www.ncbi.nlm.nih.gov/gene/11200
    
    'CIT':      'CRIK',     # https://www.ncbi.nlm.nih.gov/gene/11113
    'CSF1R':    'FMS',      # https://www.ncbi.nlm.nih.gov/gene/1436
    
    'CSNK1A1':  'CK1a',     # https://www.ncbi.nlm.nih.gov/gene/1452
    'CSNK1A1L': 'CK1',      # https://www.ncbi.nlm.nih.gov/gene/122011
    'CSNK1D':   'CK1d',     # https://www.ncbi.nlm.nih.gov/gene/1453
    'CSNK1E':   'CK1e',     # https://www.ncbi.nlm.nih.gov/gene/1454
    'CSNK1G1':  'CK1g1',    # https://www.ncbi.nlm.nih.gov/gene/53944
    'CSNK1G2':  'CK1g2',    # https://www.ncbi.nlm.nih.gov/gene/1455
    'CSNK1G3':  'CK1g3',    # https://www.ncbi.nlm.nih.gov/gene/1456
    'CSNK2A1':  'CK2a1',    # https://www.ncbi.nlm.nih.gov/gene/1457
    'CSNK2A2':  'CK2a2',    # https://www.ncbi.nlm.nih.gov/gene/1459
    
    'DCAMKL1':  'DCLK1',    # https://www.ncbi.nlm.nih.gov/gene/9201
    'DCAMKL2':  'DCLK2',    # https://www.ncbi.nlm.nih.gov/gene/166614
    'DCAMKL3':  'DCLK3',    # https://www.ncbi.nlm.nih.gov/gene/85443
    
    'EIF2AK1':  'HRI',      # https://www.ncbi.nlm.nih.gov/gene/27102
    'ERK8':     'ERK7',     # https://www.ncbi.nlm.nih.gov/gene/225689
    'ERN1':     'IRE1',     # https://www.ncbi.nlm.nih.gov/gene/2081
    
    'GRK1':     'RHOK',     # https://www.ncbi.nlm.nih.gov/gene/6011
    'GRK4':     'GPRK4',    # https://www.ncbi.nlm.nih.gov/gene/2868
    'GRK7':     'GPRK7',    # https://www.ncbi.nlm.nih.gov/gene/131890
    
    'INSRR':    'IRR',      # https://www.ncbi.nlm.nih.gov/gene/3645
    'MAP3K15':  'MAP3K2',   # https://www.ncbi.nlm.nih.gov/gene/389840 **Unsure on this one
    'MAP4K2':   'GCK',      # https://www.ncbi.nlm.nih.gov/gene/5871
    'MAP4K3':   'HGK',      # https://www.ncbi.nlm.nih.gov/gene/8491 **
    'MAP4K4':   'HGK',      # https://www.ncbi.nlm.nih.gov/gene/9448
    'MAP4K5':   'KHS1',     # https://www.ncbi.nlm.nih.gov/gene/11183
    
    'MEK1':     'Erk1',     # https://www.ncbi.nlm.nih.gov/gene/828713
    'MEK2':     'Erk2',     # *
    'MEK3':     'Erk3',     # *
    'MEK4':     'Erk4',     # *
    'MEK5':     'Erk5',     # *
    'MEK6':     'Erk7',     # *
    
    'MERTK':    'MER',      # https://www.ncbi.nlm.nih.gov/gene/10461
    'MKK7':     'MAP2K7',   # https://www.ncbi.nlm.nih.gov/gene/5609
    
    'MKNK1':    'MNK1',     # https://www.ncbi.nlm.nih.gov/gene/8569
    'MKNK2':    'MNK2',     # https://www.ncbi.nlm.nih.gov/gene/2872
    
    'MST1R':    'RON',      # https://www.ncbi.nlm.nih.gov/gene/4486
    'MTOR':     'FRAP',     # https://www.ncbi.nlm.nih.gov/gene/2475
    
    'MYLK':     'smMLCK',   # https://www.ncbi.nlm.nih.gov/gene/4638
    'MYLK2':    'skMLCK',   # https://www.ncbi.nlm.nih.gov/gene/85366
    'MYLK4':    'SgK085',   # https://www.ncbi.nlm.nih.gov/gene/340156
    
    'PAK7':     'PAK5',     # https://www.ncbi.nlm.nih.gov/gene/57144
    
    'PCTK1':    'PCTAIRE1', # https://www.ncbi.nlm.nih.gov/gene/5127
    'PCTK2':    'PCTAIRE2', # https://www.ncbi.nlm.nih.gov/gene/5128
    'PCTK3':    'PCTAIRE3', # https://www.ncbi.nlm.nih.gov/gene/5129
    
    'PDPK1':    'PDK1',     # https://www.ncbi.nlm.nih.gov/gene/5170
    'PFCDPK1':  'Other',  # https://www.ncbi.nlm.nih.gov/gene/815931 ** discontinued?
    'PFPK5':    'CDK2',     # High sequence similarity to CDK2 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6436633/)
    'PFTK1':    'PFTAIRE1', # https://www.ncbi.nlm.nih.gov/gene/5218
    
    'PIK3C2B':  'PIK3R4',   # https://www.ncbi.nlm.nih.gov/gene/5287 **
    'PIK3C2G':  'PIK3R4',   # https://www.ncbi.nlm.nih.gov/gene/5288 **
    'PIK3CA':   'PIK3R4',   # *
    'PIK3CB':   'PIK3R4',   # *
    'PIK3CD':   'PIK3R4',   # *
    'PIK3CG':   'PIK3R4',   # *
    'PIK4CB':   'PIK3R4',   # *
    
    'PIP5K1A':  'Other',    # https://www.ncbi.nlm.nih.gov/gene/8394 ** FAILED TO FIND ALIAS
    'PIP5K1C':  'Other',
    'PIP5K2B':  'Other',
    'PIP5K2C':  'Other',
    
    'PKMYT1':   'MYT1',     # https://www.ncbi.nlm.nih.gov/gene/9088
    'PKNB':     'Other',    # https://www.ncbi.nlm.nih.gov/gene/887072 ** FAILED TO FIND ALIAS
    
    'PRKCD':    'PKCd',     # https://www.ncbi.nlm.nih.gov/gene/5580
    'PRKCE':    'PKCe',     # https://www.ncbi.nlm.nih.gov/gene/5581
    'PRKCH':    'PKCh',     # https://www.ncbi.nlm.nih.gov/gene/5583
    'PRKCI':    'PKCi',     # https://www.ncbi.nlm.nih.gov/gene/5584
    'PRKCQ':    'PKCt',     # https://www.ncbi.nlm.nih.gov/gene/5588 *
    
    'PRKD1':    'PKD1',     # https://www.ncbi.nlm.nih.gov/gene/5587
    'PRKD2':    'PKD2',     # https://www.ncbi.nlm.nih.gov/gene/25865
    'PRKD3':    'PKD3',     # https://www.ncbi.nlm.nih.gov/gene/23683
    
    'PRKG1':    'PKG1',     # https://www.ncbi.nlm.nih.gov/gene/5592
    'PRKG2':    'PKG2',     # https://www.ncbi.nlm.nih.gov/gene/5593
    
    'PRKR':     'PKR',      # https://www.ncbi.nlm.nih.gov/gene/5610
    
    'RIPK4':    'ANKRD3',   # https://www.ncbi.nlm.nih.gov/gene/54101
    'RIPK5':    'ANKRD3',   # https://www.ncbi.nlm.nih.gov/gene/11035 ** FAILED TO FIND ALIAS
    
    'ROS1':     'ROS',      # https://www.ncbi.nlm.nih.gov/gene/6098
    
    'RPS6KA4':  'RSK3',     # https://www.ncbi.nlm.nih.gov/gene/8986
    'RPS6KA4':  'RSK1~b',   # https://www.ncbi.nlm.nih.gov/gene/8986
    'RPS6KA5':  'MSK1',     # https://www.ncbi.nlm.nih.gov/gene/9252
    
    'S6K1':     'p70S6K',   # https://www.ncbi.nlm.nih.gov/gene/6198
    'SBK1':     'SBK',      # https://www.ncbi.nlm.nih.gov/gene/388228
    'SIK2':     'QIK',      # https://www.ncbi.nlm.nih.gov/gene/23235
    'SNARK':    'NUAK2',    # https://www.ncbi.nlm.nih.gov/gene/81788
    'SRMS':     'SRM',      # https://www.ncbi.nlm.nih.gov/gene/6725
    'SRPK3':    'MSSK1',    # https://www.ncbi.nlm.nih.gov/gene/26576
    
    'STK16':    'MPSK1',    # https://www.ncbi.nlm.nih.gov/gene/8576
    'STK35':    'CLIK1',    # https://www.ncbi.nlm.nih.gov/gene/140901
    'STK36':    'CLIK1',    # https://www.ncbi.nlm.nih.gov/gene/27148 ** FAILED TO FIND ALIAS
    'STK39':    'PASK',     # https://www.ncbi.nlm.nih.gov/gene/27347
    
    'TAOK1':    'TAO1',     # https://www.ncbi.nlm.nih.gov/gene/57551
    'TAOK2':    'TAO2',     # https://www.ncbi.nlm.nih.gov/gene/9344
    'TAOK3':    'TAO3',     # https://www.ncbi.nlm.nih.gov/gene/51347
    
    'TNK2':     'ACK',      # https://www.ncbi.nlm.nih.gov/gene/10188
    'TNNI3K':   'p38a',     # https://www.ncbi.nlm.nih.gov/gene/51208 ** FAILED TO FIND ALIAS
    
    'TRPM6':    'ChaK2',    # https://www.ncbi.nlm.nih.gov/gene/140803
    'TSSK1B':   'TSSK1',    # https://www.ncbi.nlm.nih.gov/gene/83942
    'VEGFR2':   'KDR',      # https://www.ncbi.nlm.nih.gov/gene/3791
    'WEE2':     'Wee1B',    # https://www.ncbi.nlm.nih.gov/gene/494551
    'YSK4':     'MAP3K1',  # https://www.ncbi.nlm.nih.gov/gene/80122 ** FAILED TO FIND ALIAS
}

# %% merging missing proteins with main df
# merge on index or just assign as main_family if not found

missing_prot_merged = {}
for k, name in missing_prots.items():
    matches = df.index[df.index.str.lower() == name.lower()]
    
    if len(matches) > 0:
        name = matches[0]
        missing_prot_merged[k] = (df.loc[name, 'main_family'], df.loc[name, 'subgroup'], None)
    else:
        missing_prot_merged[k] = (name, None, None)

# %%
missing_df = pd.DataFrame.from_dict(missing_prot_merged, orient='index', 
                                    columns=df.columns)

#%%
df = pd.concat([df, missing_df])

#%%

prot_dict = json.load(open(f'{cfg.DATA_ROOT}/davis/proteins.txt', 'r'))
prots = check_davis_names(prot_dict, df)


# %% plot histogram of main families and their counts
main_families = [v[1] for v in prots.values()]
main_families = pd.Series(main_families)
sns.set_theme(style='darkgrid')
plt.figure(figsize=(10, 5))
sns.histplot(main_families)
plt.xlabel('Protein Kinase Family')
plt.title('Distribution of Protein Kinase Families in Davis Dataset')
plt.ylabel('Count (442 total proteins)')
plt.savefig('results/figures/davis_kinaseFamilies.png', dpi=300, bbox_inches='tight')

# %%
