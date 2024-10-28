import os, logging, argparse
parser = argparse.ArgumentParser(description='Runs Mutagenesis on an input PDB file and a given ligand SMILES.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-ls', '--ligand_smiles', type=str, help='Ligand SMILES string.')
group.add_argument('-sdf', '--ligand_sdf', type=str, help='File path to SDF file (needed for GVPL features).')

parser.add_argument('--ligand_id', type=str, required=True, help='Ligand SMILES identifier, required for output path.')
parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file.')
parser.add_argument('--out_path', type=str, default='./', 
                    help='Output directory path to save resulting mutagenesis numpy matrix with predicted pkd values')
parser.add_argument('--res_start', type=int, default=0, help='Start index for mutagenesis (zero-indexed).')
parser.add_argument('--res_end', type=int, default=float('inf'), help='End index for mutagenesis.')

parser.add_argument('--model_opt', type=str, default='davis_DG', 
                    choices=['davis_DG',    'davis_gvpl',   'davis_esm', 
                             'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                             'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
                             'PDBbind_gvpl_aflow'],
                    help='Model option. See MutDTA/src/__init__.py for details.')
parser.add_argument('--fold', type=int, default=1, 
                    help='Which model fold to use (there are 5 models for each option due to 5-fold CV).')
parser.add_argument("-D", "--only_download", help="for downloading esm models if the are missing", default=False, action="store_true")
args = parser.parse_args()

# Assign variables
LIGAND_SMILES = args.ligand_smiles
LIGAND_SDF = args.ligand_sdf
if LIGAND_SDF:
    from rdkit import Chem
    LIGAND_SMILES = Chem.MolToSmiles(Chem.MolFromMolFile(LIGAND_SDF))

LIGAND_ID = args.ligand_id
PDB_FILE = args.pdb_file
OUT_PATH = args.out_path
MODEL_OPT = args.model_opt
FOLD = args.fold
RES_START = args.res_start
RES_END = args.res_end

OUT_DIR = f'{OUT_PATH}/{LIGAND_ID}/{MODEL_OPT}'
ONLY_DOWNLOAD = args.only_download
os.makedirs(OUT_DIR, exist_ok=True)

logging.getLogger().setLevel(logging.DEBUG)
print("#"*50)
print(f"LIGAND_SMILES: {LIGAND_SMILES}")
print(f"LIGAND_ID: {LIGAND_ID}")
print(f"PDB_FILE: {PDB_FILE}")
print(f"OUT_PATH: {OUT_PATH}")
print(f"MODEL_OPT: {MODEL_OPT}")
print(f"FOLD: {FOLD}")
print(f"RES_START: {RES_START}")
print(f"RES_END: {RES_END}")
print(f"OUT_DIR: {OUT_DIR}")
print(f"ONLY_DOWNLOAD: {ONLY_DOWNLOAD}")
print("#"*50, end="\n\n")


import numpy as np
import torch
import torch_geometric as torchg
from tqdm import tqdm

from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.utils.residue import ResInfo
from src.data_prep.quick_prep import get_ligand_features, get_protein_features

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = TUNED_MODEL_CONFIGS[MODEL_OPT]
PDB_FILE_NAME = os.path.basename(PDB_FILE).split('.pdb')[0]

##################################################
### Loading the model and get original pkd value #
##################################################
MODEL, _ = Loader.load_tuned_model(MODEL_OPT, fold=FOLD, device=DEVICE)
MODEL.eval()
print(f"MODEL LOADED - {MODEL.__class__}")

if ONLY_DOWNLOAD:
    logging.critical("ONLY DOWNLOAD OPTION SET, EXITING")
    exit()

# build ligand graph
mol_feat, mol_edge = get_ligand_features(LIGAND_SMILES, MODEL_PARAMS['lig_feat_opt'], 
                               MODEL_PARAMS['lig_edge_opt'], LIGAND_SDF)
lig = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge), lig_seq=LIGAND_SMILES)

# build protein graph
pro, pdb_original = get_protein_features(PDB_FILE)
original_seq = pdb_original.sequence

original_pkd = MODEL(pro.to(DEVICE), lig.to(DEVICE))
print("Original pkd:", original_pkd)


##################################################
### Mutate and regenerate graphs #################
##################################################
# zero indexed res range to mutate:
res_range = (max(RES_START, 0), min(RES_END, len(original_seq)))

from src.utils.mutate_model import run_modeller
amino_acids = ResInfo.amino_acids[:-1] # not including "X" - unknown
muta = np.zeros(shape=(len(amino_acids), len(original_seq)))

with tqdm(range(*res_range), ncols=100, total=(res_range[1]-res_range[0]), desc='Mutating') as t:
    for j in t:
        for i, AA in enumerate(amino_acids):
            if i%2 == 0:
                t.set_postfix(res=j, AA=i+1)
            
            if original_seq[j] == AA: # skip same AA modifications 
                muta[i,j] = original_pkd
                continue
            out_pdb_fp = run_modeller(PDB_FILE, j+1, ResInfo.code_to_pep[AA], "A")
            
            pro, _ = get_protein_features(out_pdb_fp)
            assert pro.pro_seq != original_seq and pro.pro_seq[j] == AA, \
                f"ERROR in modeller, {pro.pro_seq} == {original_seq} \nor {pro.pro_seq[j]} != {AA}"
            
            muta[i,j] = MODEL(pro.to(DEVICE), lig.to(DEVICE))
            
            # delete after use
            os.remove(out_pdb_fp)


# Save mutagenesis matrix
OUT_FP = f"{OUT_DIR}/{res_range[0]}_{res_range[1]}.npy"
print("Saving mutagenesis numpy matrix to", OUT_FP)
np.save(OUT_FP, muta)