#%%
import os, re, argparse
from utils import split_structure, get_df, plot_together
debug = True
#%%
parser = argparse.ArgumentParser(description='Split pdbqt file into protein and ligand')
parser.add_argument('-r', metavar='--receptor', type=str, help='path to pdbqt file', 
                    required=True)
parser.add_argument('-a', help='To extract all structures',
                    required=False, default=False, action='store_true')

args =parser.parse_args(['-r', 'test_prep/prep/1a1e.pdbqt']) if debug else  parser.parse_args()
#%%
r_path = '/'.join(args.r.split('/')[:-1])

os.makedirs(r_path, exist_ok=True)
split_structure(file_path=args.r, save='all' if args.a else 'mains')

res = 'y' if debug else input('Validate/view protein and ligand (Y/N)?')
if res.lower() == 'y':
    dfP, dfL = None, None
    for f_name in os.listdir(r_path):
        if re.search(r'split[\w\-]*.pdbqt',f_name):
            if re.search('ligand',f_name):
                dfL = get_df(open(f'{r_path}/{f_name}','r').readlines())
            else:
                dfP = get_df(open(f'{r_path}/{f_name}','r').readlines())
                
    if dfP is not None and dfL is not None:
        plot_together(dfL,dfP)
    else:
        print('Split failed, missing protein or ligand file')
        exit(1)
# %%
