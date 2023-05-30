"""
This creates the configuration file with a focus on the binding site.

Using ligand position to identify where the binding region is.

# NOTE: Binding info is provided during split_pdb step instead 
# this way we dont have to keep ligand pdb since it is not good for docking.
"""

import argparse, os

parser = argparse.ArgumentParser(description='Prepares config file for AutoDock Vina.')
parser.add_argument('-p', metavar='--prep_path', type=str,
                    help="Directory containing prepared ligand and protein. \
                    With file names ending in 'ligand.pdqt' or 'receptor.pdqt'.", required=False)
parser.add_argument('-r', metavar='--receptor', type=str, 
                    help='Path to pdbqt file containing sole protein.', required=False)
parser.add_argument('-l', metavar='--ligand', type=str, 
                    help='Path to pdbqt file containing sole ligand.', required=False)
parser.add_argument('-o', metavar='--output', type=str,
                    help='Output config file path. Default is to save it \
                    in the same location as the receptor as "conf.txt"', required=False)

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
        
    
    conf = { # These are default values set by AutoDock Vina (see: https://vina.scripps.edu/manual/#config)
        "receptor": args.r,
        "ligand": args.l,
        "energy_range": 3,   # maximum energy difference between the best binding mode and the worst one (kcal/mol)
        "exhaustiveness": 8, # exhaustiveness of the global search (roughly proportional to time)
        "num_modes": 9,      # maximum number of binding modes to generate
        #"cpu": 1,           # num cpus to use. Default is to automatically detect.
    }

    # saving config file
    if args.o is None:
        args.o = '/'.join(conf["receptor"].split('/')[:-1]) + '/conf.txt'
        
    with open(args.o, 'a') as f:
        for key, value in conf.items():
            f.write(f'{key} = {value}\n')