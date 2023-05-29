import os, re, argparse
from helpers.format_pdb import split_structure, get_df
from helpers.display import plot_together

parser = argparse.ArgumentParser(description='Split pdbqt file into protein and ligand')
parser.add_argument('-r', metavar='--receptor', type=str, help='path to pdbqt file', 
                    required=True)
parser.add_argument('-v', help='To validate and view protein and ligand',
                    required=False, default=False, action='store_true')

parser.add_argument('-s', metavar='--save', type=str, 
                    help='What structures to save ("all", "mains", or "largest")',
                    required=False, default='mains')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.s.lower() not in ['all', 'mains', 'largest']:
        parser.error(f'Invalid save option: {parser.parse_args().s}')

    r_path = '/'.join(args.r.split('/')[:-1])

    # making sure save path exists
    os.makedirs(r_path, exist_ok=True)
    # splitting pdbqt file
    split_structure(file_path=args.r, save=args.s.lower())

    # viewing protein and ligand if requested
    if args.v == 'y':
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
