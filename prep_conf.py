import argparse, os
from utils import get_df

parser = argparse.ArgumentParser(description='Prepares config file for AutoDock Vina.')
parser.add_argument('-p', metavar='--prep_path', type=str,
                    help='Prep directory for split ligand and protein.', required=False)
parser.add_argument('-r', metavar='--receptor', type=str, 
                    help='Path to pdbqt file containing sole protein.', required=False)
parser.add_argument('-l', metavar='--ligand', type=str, 
                    help='Path to pdbqt file containing sole ligand.', required=False)
parser.add_argument('-o', metavar='--output', type=str,
                    help='Output config file path. Default is in same dir as receptor as "conf.txt"', required=False)

if __name__ == '__main__':
    args = parser.parse_args()

    # (args.p is None) implies (args.r is not None and args.l is not None)
    # !p -> (r&l)
    # p || (r&l)
    if not((args.p is not None) or (args.r is not None and args.l is not None)):
        parser.error('Either prep_path or receptor and ligand must be provided.')
        
    if args.p is not None:
        # Automatically finding protein and ligand files
        # should be named *_receptor.pdbqt and *_ligand.pdbqt if created by split_pdb.py
        for file in os.listdir(args.p):
            if file.endswith('receptor.pdbqt'):
                args.r = f'{args.p}/{file}'
            elif file.endswith('ligand.pdbqt'):
                args.l = f'{args.p}/{file}'
        
    
    conf = { # These are default values set by AutoDock Vina
        "receptor": args.r,
        "ligand": args.l,
        "energy_range": 3,   # maximum energy difference between the best binding mode and the worst one (kcal/mol)
        "exhaustiveness": 8, # exhaustiveness of the global search (roughly proportional to time)
        "num_modes": 9,      # maximum number of binding modes to generate
        #"cpu": 1,           # Automatic detection. Default is 1.
    }

    lig = get_df(open(args.l,'r').readlines())
    prot = get_df(open(args.r,'r').readlines())[['x', 'y', 'z']]

    # Getting ligand center
    # note PDB units are in angstroms (see: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html)
    lig_center = lig[['x', 'y', 'z']].mean().values # center of mass

    # Making sure it is inside the protein
    pxmin, pxmax = prot['x'].min(), prot['x'].max()
    pymin, pymax = prot['y'].min(), prot['y'].max()
    pzmin, pzmax = prot['z'].min(), prot['z'].max()
    assert (pxmin < lig_center[0] < pxmax and
            pymin < lig_center[1] < pymax and
            pzmin < lig_center[2] < pzmax), "Ligand center is not inside protein."

    conf["center_x"] = lig_center[0]
    conf["center_y"] = lig_center[1]
    conf["center_z"] = lig_center[2]

    # Getting box size and padding by 20A
    conf["size_x"] = (prot['x'].max() - prot['x'].min())/2 + 20
    conf["size_y"] = (prot['y'].max() - prot['y'].min())/2 + 20
    conf["size_z"] = (prot['z'].max() - prot['z'].min())/2 + 20

    # saving config file
    if args.o is None:
        args.o = '/'.join(conf["receptor"].split('/')[:-1]) + '/conf.txt'
        
    with open(args.o, 'w') as f:
        for key, value in conf.items():
            f.write(f'{key} = {value}\n')
#%%