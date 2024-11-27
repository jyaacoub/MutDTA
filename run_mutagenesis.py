import os, logging, argparse
parser = argparse.ArgumentParser(description='Runs Mutagenesis on an input PDB file and a given ligand SMILES.')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-ls', '--ligand_smiles', type=str, help='Ligand SMILES string.')
group.add_argument('-sdf', '--ligand_sdf', type=str, help='File path to SDF file (needed for GVPL features).')
parser.add_argument('--ligand_id', type=str, required=True, 
                    help='Ligand SMILES identifier, required for creating unique output path.')

parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file.')
parser.add_argument('--out_path', type=str, default='./', 
                    help='Output directory path to save resulting mutagenesis numpy matrix with predicted pkd values')
parser.add_argument('-na', '--num_modeller_attempts', type=int, required=False, default=5, 
                    help='Number of attempts for modeller to resolve steric clashes. If it cant resolve it within'+ \
                        ' this number of attempts it will set to np.nan. Defaults to 5.')

full_mut = parser.add_argument_group('FULL SATURATION MUTAGENESIS ARGS', 
                                     description="This is the default unless mutations are specified")
full_mut.add_argument('--res_start', type=int, default=0, help='Start index for mutagenesis (zero-indexed).')
full_mut.add_argument('--res_end', type=int, default=float('inf'), help='End index for mutagenesis.')

partial_mut = parser.add_argument_group('PARTIAL MUTAGENESIS ARGS',
                                           description="Less intensive for when a full staturation is not needed. " + \
                                               "Runs inference twice - once on native and once on mutated structure.")
partial_mut.add_argument('-mut', '--mutations',  type=str, nargs='+', required=False,
                       help="The mutations to apply to the native structure in the format <native AA><index><mut AA> "+\
                           "(e.g.: M230A). Note that index starts as 1 as per PDB documentation.")

model_args = parser.add_argument_group('MODEL ARGS')
model_args.add_argument('--model_opt', type=str, default='davis_DG', 
                    choices=['davis_DG',    'davis_gvpl',   'davis_esm', 
                             'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                             'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
                             'PDBbind_gvpl_aflow'],
                    help='Model option. See MutDTA/src/__init__.py for details.')
model_args.add_argument('--fold', type=int, default=1, 
                    help='Which model fold to use (there are 5 models for each option due to 5-fold CV).')
model_args.add_argument("-D", "--only_download", help="for downloading esm models if the are missing", default=False, action="store_true")
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
MUTATIONS=args.mutations
NUM_MODELLER_ATTEMPTS = args.num_modeller_attempts

OUT_DIR = f'{OUT_PATH}/{LIGAND_ID}/{MODEL_OPT}'
ONLY_DOWNLOAD = args.only_download

logging.getLogger().setLevel(logging.DEBUG)
print("#"*50)
print(f"LIGAND_SMILES: {LIGAND_SMILES}")
print(f"   LIGAND_SDF: {LIGAND_SDF}")
print(f"    LIGAND_ID: {LIGAND_ID}")
print(f"     PDB_FILE: {PDB_FILE}")
print(f"     OUT_PATH: {OUT_PATH}")
print(f"      OUT_DIR: {OUT_DIR}")

print(f"\n    RES_START: {RES_START}")
print(f"      RES_END: {RES_END}")

print(f"\n    MUTATIONS: {MUTATIONS}")

print(f"\n    MODEL_OPT: {MODEL_OPT}")
print(f"         FOLD: {FOLD}")
print(f"ONLY_DOWNLOAD: {ONLY_DOWNLOAD}")
print("#"*50, end="\n\n")
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import torch
from tqdm import tqdm

from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.utils.residue import ResInfo
from src.data_prep.quick_prep import get_ligand_features, get_protein_features
from src.utils.mutate_model import run_modeller_multiple, run_modeller


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
lig = get_ligand_features(LIGAND_SMILES, MODEL_PARAMS['lig_feat_opt'], MODEL_PARAMS['lig_edge_opt'], LIGAND_SDF)

# build protein graph
pro, pdb_original = get_protein_features(PDB_FILE, MODEL_PARAMS['feature_opt'], MODEL_PARAMS['edge_opt'])
original_seq = pdb_original.sequence

original_pkd = MODEL(pro.to(DEVICE), lig.to(DEVICE))
print("Original pkd:", original_pkd, end="\n\n")

if MUTATIONS:
    mut_pdb_file = run_modeller_multiple(PDB_FILE, MUTATIONS, n_attempts=NUM_MODELLER_ATTEMPTS)
    print(mut_pdb_file)
    pro, _ = get_protein_features(mut_pdb_file, MODEL_PARAMS['feature_opt'], MODEL_PARAMS['edge_opt'])
    mut_pkd = MODEL(pro.to(DEVICE), lig.to(DEVICE))
    print("\nMutated pkd:", mut_pkd)
else:
    logging.warning("No mutations were passed in - running full saturation mutagenesis")
    # zero indexed res range to mutate:
    res_range = (max(RES_START, 0), min(RES_END, len(original_seq)))
    
    amino_acids = ResInfo.amino_acids[:-1] # not including "X" - unknown
    muta = np.zeros(shape=(len(amino_acids), len(original_seq)))

    with tqdm(range(*res_range), ncols=100, total=(res_range[1]-res_range[0]), 
              desc='Saturation mutagenesis') as t:
        for j in t:
            for i, AA in enumerate(amino_acids):
                if i%2 == 0:
                    t.set_postfix(res=j, AA=i+1)
                
                if original_seq[j] == AA: # skip same AA modifications 
                    muta[i,j] = original_pkd
                    continue
                try:
                    out_pdb_fp = run_modeller(PDB_FILE, j+1, ResInfo.code_to_pep[AA], "A", n_attempts=NUM_MODELLER_ATTEMPTS)
                except OverflowError as e:
                    muta[i,j] = np.NAN
                    continue
                
                pro, _ = get_protein_features(out_pdb_fp, MODEL_PARAMS['feature_opt'], MODEL_PARAMS['edge_opt'])
                assert pro.pro_seq != original_seq and pro.pro_seq[j] == AA, \
                    f"ERROR in modeller, {pro.pro_seq} == {original_seq} \nor {pro.pro_seq[j]} != {AA}"
                
                muta[i,j] = MODEL(pro.to(DEVICE), lig.to(DEVICE))
                
                # delete after use
                os.remove(out_pdb_fp)


    # Save mutagenesis matrix
    OUT_FP = f"{OUT_DIR}/{res_range[0]}_{res_range[1]}-{os.path.basename(PDB_FILE).split('.pdb')[0]}.npy"
    print("Saving mutagenesis numpy matrix to", OUT_FP)
    np.save(OUT_FP, muta)