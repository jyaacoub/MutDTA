import os, argparse, re, math
from tqdm import tqdm

# Parse and extract args
parser = argparse.ArgumentParser(description='Extracts vina results from out files. \
    (Example use: \
    python extract_vina_out.py ./data/refined-set/ data/vina_out/run3.csv -sl ./data/shortlists/no_err_50/sample.csv)')
parser.add_argument('path', type=str, 
                    help='Path to PDBbind dir containing pdb for protein to convert to pdbqt or simple dir containing just vina outs (see arg -fr).')
parser.add_argument('out_csv', type=str, 
                    help='Output path for the csv file (e.g.: PATH/TO/FILE/vina_out.csv)')
parser.add_argument('-sl', metavar='--shortlist', type=str, required=False,
                    help='Shortlist csv file containing pdbcodes to extract. Otherwise extracts all in PDBbind dir.')
parser.add_argument('-dm', required=False, action='store_true', default=False,
                    help="(Dont Move) - Don't move vina_log and vina_out files to new dir with same name as out_csv file.")
parser.add_argument('-fr', required=False, action='store_true', default=False,
                    help="(From Run) - Extract data from a run directory containing all vina outs in a single dir.")
args = parser.parse_args()

vina_dir = args.path
out_csv = args.out_csv
print(f'vina_dir: {vina_dir}')
print(f'out_csv: {out_csv}')
print(f'sl: {args.sl}')
if args.sl is None:
    # Looping through all the pdbcodes in the PDBbind dir, select everything except index and readme dir
    codes = [d for d in os.listdir(vina_dir) if os.path.isdir(os.path.join(vina_dir, d)) and d not in ["index", "readme"]]
else:
    with open(args.sl, 'r') as f:
        codes = [line.split(',')[0] for line in f.readlines()]

if not args.dm:
    new_dir = os.path.join(os.path.dirname(out_csv), os.path.basename(out_csv).split('.')[0])
    print(f'new dir: {new_dir}')
    os.mkdir(new_dir)

# Getting count of dirs
total = len(codes)

count = 0
errors = 0

R=0.0019870937 # kcal/Mol*K (gas constant)
T=273.15       # K
RT = R*T

with open(out_csv, "w") as out_f:
    out_f.write("PDBCode,vina_deltaG(kcal/mol),vina_kd(uM)\n")

    for code in tqdm(codes, desc='Extracting affinity values'):
        dir_path = vina_dir if args.fr else os.path.join(vina_dir, code)
        log = os.path.join(dir_path, f"{code}_vina_log.txt")
        
        if not os.path.isfile(log):
            errors += 1
            print(log)
            continue

        with open(log, "r") as f:
            pattern = r'\s*[0-9]+\s+([-]?[0-9]+\.[0-9]+)' 
            # matching first mode only
            deltaG = re.search(pattern, f.read()).group(1)
            deltaG = float(deltaG)
            
            # converting to kd
            kd = math.e**(deltaG/RT) # Kd = e^((deltaG)/RT) # M
            kd *= 1e6                # uM = u mol/L

            out_f.write(f"{code},{deltaG},{kd}\n")
        
        # also moving vina log and out files
        if not args.dm:
            os.rename(src=os.path.join(dir_path, f"{code}_vina_out.pdbqt"), 
                      dst=os.path.join(new_dir, f"{code}_vina_out.pdbqt"))
            os.rename(src=log, 
                      dst=os.path.join(new_dir, f"{code}_vina_log.txt"))
            
        
        count += 1

print(f"Total: {total} dirs processed")
print(f"Errors: {errors}")