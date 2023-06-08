"""
This creates the configuration file with a focus on the binding site.

Using ligand position to identify where the binding region is.

# NOTE: Binding info is provided during split_pdb step instead 
# this way we dont have to keep ligand pdb since it is not good for docking.
"""

import argparse, os, re
from os import path as op
from helpers.format_pdb import get_coords

parser = argparse.ArgumentParser(description='Prepares config file for AutoDock Vina.')
parser.add_argument('-p', metavar='--prep_path', type=str,
                    help="Directory containing prepared ligand and protein. \
                    With file names ending in 'ligand.pdqt' or 'receptor.pdqt'.", required=False)

parser.add_argument('-r', metavar='--receptor', type=str, 
                    help='Path to pdbqt file containing sole protein.', required=False)
parser.add_argument('-l', metavar='--ligand', type=str, 
                    help='Path to pdbqt file containing sole ligand.', required=False)

parser.add_argument('-pp', metavar='--pocket_path', type=str,
                    help='binding pocket pdb file from PDBbind', required=False)

parser.add_argument('-o', metavar='--output', type=str,
                    help='Output config file path. Default is to save it \
                    in the same location as the receptor as "conf.txt"', required=False)
parser.add_argument('-c', metavar='--conf_path', type=str,
                    help='Path to config file to use as template. Default is to use \
                        AutoDock Vina defaults (see: https://vina.scripps.edu/manual/#config).', 
                        required=False)

parser.add_argument('-pdb', metavar='pdbcode', type=str, required=False,
                    help='PDBcode to use for naming out and log files. Default is to extract \
                        it from the receptor file name (assuming it is named as "<code>*.pdbqt")')

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
                args.r = op.join(args.p, file)
            elif file.endswith('ligand.pdbqt'):
                args.l = op.join(args.p, file)
                
    if args.pdb is None:
        # extracting from receptor file name
        # these codes are always 4 char in length with the first being a numerical number 
        # (see docs: https://proteopedia.org/wiki/index.php/PDB_code)
        # pdbcode HAS to be the first part of the file name
        # i.e.: grep translation - <PDBcode>*.pdbqt
        p = r"([0-9]+[0-9A-Za-z]{3})[_A-z]*\.pdbqt$" 
        PDBcode = re.search(p, op.basename(args.r), re.MULTILINE).group(1)
    else:
        PDBcode = args.pdb
    
    conf = {
        # These are the default values set by AutoDock Vina (see: https://vina.scripps.edu/manual/#config)
        "energy_range": 3,   # maximum energy difference between the best binding mode and the worst one (kcal/mol)
        "exhaustiveness": 8, # exhaustiveness of the global search (roughly proportional to time)
        "num_modes": 9,      # maximum number of binding modes to generate
        #"cpu": 1,           # num cpus to use. Default is to automatically detect.
        
        # My defaults (output files are sent to same place as receptor pdbqt file)
        'receptor': args.r,
        'ligand': args.l,
        "out": op.dirname(args.r),
        "log": op.dirname(args.r),
        "seed": 904455071,
    }
    
    if args.c is not None:
        # Defaults are overwritten if provided in conf template
        with open(args.c, 'r') as f:
            for line in f.readlines():
                k, v = line.split('=')
                conf[k.strip()] = v.strip()
    
    # adding file name to out paths
    conf['out'] = op.join(conf['out'], f'{PDBcode}_vina_out.pdbqt')
    conf['log'] = op.join(conf['log'], f'{PDBcode}_vina_log.txt')
    
    # saving binding site info if path provided
    if args.pp is not None:
        pocket_df = get_coords(open(args.pp, 'r').readlines())
        conf["center_x"] = pocket_df["x"].mean()
        conf["center_y"] = pocket_df["y"].mean()
        conf["center_z"] = pocket_df["z"].mean()
        conf["size_x"] = pocket_df["x"].max() - pocket_df["x"].min()
        conf["size_y"] = pocket_df["y"].max() - pocket_df["y"].min()
        conf["size_z"] = pocket_df["z"].max() - pocket_df["z"].min()    

    # saving config file in same path as receptor
    if args.o is None:
        args.o = op.join(op.dirname(args.r), f'/{PDBcode}_conf.txt')
        
    with open(args.o, 'w') as f:
        for key, value in conf.items():
            f.write(f'{key} = {value}\n')
        