import os, argparse, re, math
from tqdm import tqdm

# Parse and extract args
parser = argparse.ArgumentParser(description='Extracts vina results from out files')
parser.add_argument('path', type=str, 
                    help='path to PDBbind dir containing pdb for protein to convert to pdbqt')
parser.add_argument('out_csv', type=str, help='output path to csv file')
args = parser.parse_args()

PDBbind_dir = args.path
out_path = args.out_csv


# Looping through all the pdbcodes in the PDBbind dir, select everything except index and readme dir
dirs = [d for d in os.listdir(PDBbind_dir) if os.path.isdir(os.path.join(PDBbind_dir, d)) and d not in ["index", "readme"]]
# Getting count of dirs
total = len(dirs)

count = 0
errors = 0

R=0.0019870937 # kcal/Mol*K (gas constant)
T=273.15       # K
RT = R*T

with open(out_path, "w") as out_f:
    out_f.write("PDBCode,deltaG(kcal/mol),kd(uM)\n")

    for code in tqdm(dirs, desc='Extracting affinity values'):
        dir_path = os.path.join(PDBbind_dir, code)
        out = os.path.join(dir_path, f"{code}_vina_log.txt")
        
        if not os.path.isfile(out):
            errors += 1
            continue

        try:
            with open(out, "r") as f:
                pattern = r'\s*[0-9]+\s+([-]?[0-9]+\.[0-9]+)' 
                # matching first mode only
                deltaG = re.search(pattern, f.read()).group(1)
                deltaG = float(deltaG)
                
                # converting to kd
                kd = math.e**(deltaG/RT) # Kd = e^((deltaG)/RT) # M
                kd *= 1e6                # uM = u mol/L

                out_f.write(f"{code},{deltaG},{kd}\n")
        except ValueError as e:
            errors += 1
            print(e)
            print(f'Error on {code}. Affinity "{deltaG}" is not float')
            exit(1)

        count += 1

print(f"Total: {total} dirs processed")
print(f"Errors: {errors}")