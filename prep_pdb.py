#%% prepares files for docking given a pdbcode
import os, re
from utils import split_structure, get_df, plot_together

#%%
pdbcode = '1a1e'
path = 'test_prep/'
pdb_path = f'{path}/{pdbcode}.pdb'
prep_path = f'{path}/prep/{pdbcode}'

# checking to see that file exists
assert os.path.isfile(pdb_path), f'File {pdb_path} does not exist'

#%% run prepare_receptor4.py from cmd line
# Note that I do not use -e flag in order to extract possible ligand structures
#IMPORTANT make sure .bashrc is set up correctly:
#   alias prep_prot='pythonsh prepare_receptor4.py'
os.system(f'prep_prot -r {pdb_path} -o {prep_path}/{pdbcode}.pdbqt')

#%% split pdbqt file into separate structures (protein and ligand)
os.makedirs(prep_path, exist_ok=True)
split_structure(f'{prep_path}/{pdbcode}.pdbqt', save='mains')

#%% confirming that the split worked by visualizing them
dfP, dfL = None, None
for f_name in os.listdir(f'{prep_path}'):
    if re.search('split',f_name):
        if re.search('ligand',f_name):
            dfL = get_df(open(f'{pdbcode}/{f_name}','r').readlines())
        else:
            dfP = get_df(open(f'{pdbcode}/{f_name}','r').readlines())
            
if dfP and dfL:
    plot_together(dfL,dfP)
else:
    print('Split failed, missing protein or ligand file')