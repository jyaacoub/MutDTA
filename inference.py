import os, logging, argparse
parser = argparse.ArgumentParser(description='Runs inference on a given ligand SMILES and an input of pdb files.')


# Create a mutually exclusive group for `--ligand_smiles` and `--ligand_sdf`
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-ls', '--ligand_smiles', type=str, help='Ligand SMILES string.')
group.add_argument('-sdf', '--ligand_sdf', type=str, help='File path to SDF file for GVPL features.')


parser.add_argument('-pdb','--pdb_files', type=str, nargs='+', required=True, 
                    help='List of paths to the PDB files. This can be the alphaflow output files with the first model for edges. '+\
                        'NOTE: file name is used for protein ID in csv output')

parser.add_argument('-o','--csv_out', type=str, default='./predicted_pkd_values.csv', 
                    help='Output csv to save the predicted pkd values with the following columns: \n'+\
                        'TIMESTAMP, model, pdb_file, ligand_id, pred_pkd, SMILES, pro_seq')

parser.add_argument('-m','--model_opt', type=str, default='davis_DG', 
                    choices=['davis_DG',    'davis_gvpl',   'davis_esm', 
                             'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                             'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl', 
                             'PDBbind_gvpl_aflow'],
                    help='Model option. See MutDTA/src/__init__.py for details on hyperparameters.'+\
                        'NOTE for _aflow models we expect the pdb_file to contain multiple conformations')
parser.add_argument('-f','--fold', type=int, default=1, 
                    help='Which model fold to use (there are 5 models for each option due to 5-fold CV).')

parser.add_argument('-li','--ligand_id', type=str, default='', 
                    help='Optional identifier for ligand to save in csv output.')
parser.add_argument('-bs','--batch_size', type=int, default=8, 
                    help='Batch size for processing the PDB files. Default is set to a conservative 8 batch size '+\
                         'since that is the max a100s can comfortably support for our largest models (ESM models).')
parser.add_argument("-D", "--only_download", help="for downloading esm models if the are missing", default=False, action="store_true")
args = parser.parse_args()

# Assign variables
LIGAND_SMILES = args.ligand_smiles
LIGAND_SDF = args.ligand_sdf
if LIGAND_SDF:
    from rdkit import Chem
    LIGAND_SMILES = Chem.MolToSmiles(Chem.MolFromMolFile(LIGAND_SDF))

LIGAND_ID = args.ligand_id
PDB_FILES = args.pdb_files
CSV_OUT = args.csv_out
MODEL_OPT = args.model_opt
FOLD = args.fold
BATCH_SIZE = args.batch_size
ONLY_DOWNLOAD = args.only_download

print("#"*50)
print(f"LIGAND_SMILES: {LIGAND_SMILES}")
print(f"LIGAND_SDF: {LIGAND_SDF}")
print(f"LIGAND_ID: {LIGAND_ID}")
print(f"PDB_FILES: {PDB_FILES}")
print(f"OUT_PATH: {CSV_OUT}")
print(f"MODEL_OPT: {MODEL_OPT}")
print(f"FOLD: {FOLD}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"ONLY_DOWNLOAD: {ONLY_DOWNLOAD}")
print("#"*50, end="\n\n")
if ONLY_DOWNLOAD: logging.warning("ONLY_DOWNLOAD option set")

import pandas as pd
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from src import TUNED_MODEL_CONFIGS
from src.utils.loader import Loader
from src.data_prep.quick_prep import get_ligand_features, get_protein_features

print(f"Module imports completed")
logging.getLogger().setLevel(logging.DEBUG)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = TUNED_MODEL_CONFIGS[MODEL_OPT]

##################################################
### Loading the model                          ###
##################################################
MODEL, _ = Loader.load_tuned_model(MODEL_OPT, fold=FOLD, device=DEVICE)
MODEL.eval()
print(f"MODEL LOADED - {MODEL.__class__}")

if ONLY_DOWNLOAD:
    logging.critical("ONLY DOWNLOAD OPTION SET, EXITING")
    exit()

# build ligand graph
lig_data = get_ligand_features(LIGAND_SMILES, MODEL_PARAMS['lig_feat_opt'], 
                               MODEL_PARAMS['lig_edge_opt'], LIGAND_SDF)

# Prepare to collect results
results = []

# Process PDB files in batches
num_pdb_files = len(PDB_FILES)
num_batches = (num_pdb_files + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division

for batch_idx in tqdm(range(num_batches), desc="Running inference on PDB file(s)"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, num_pdb_files)
    batch_pdb_files = PDB_FILES[start_idx:end_idx]
    batch_pdb_file_names = [os.path.basename(pdb_file).split('.pdb')[0] for pdb_file in batch_pdb_files]

    tqdm.write(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_pdb_files)} PDB files.")

    # Build protein graphs for the current batch
    pro_list = []
    for PDB_FILE in batch_pdb_files:
        pro, _ = get_protein_features(PDB_FILE, MODEL_PARAMS['feature_opt'], MODEL_PARAMS['edge_opt'])
        pro_list.append(pro)

    # Replicate ligand data for the current batch size
    lig_list = [lig_data] * len(pro_list)

    # Batch the protein and ligand data
    pro_batch = Batch.from_data_list(pro_list)
    lig_batch = Batch.from_data_list(lig_list)

    # Move data to device
    pro_batch = pro_batch.to(DEVICE)
    lig_batch = lig_batch.to(DEVICE)

    # Run the model on the batched data
    with torch.no_grad():
        predicted_pkd = MODEL(pro_batch, lig_batch)

    # Collect results
    predicted_pkd = predicted_pkd.cpu().numpy().flatten()
    time_stamp = pd.Timestamp("now")
    for pdb_file_name, pkd_value, sq in zip(batch_pdb_file_names, predicted_pkd, pro_batch.pro_seq):
        results.append({
            'TIMESTAMP': time_stamp,
            'model': MODEL_OPT,
            'pdb_file': pdb_file_name,
            'ligand_id': LIGAND_ID,
            'pred_pkd': round(pkd_value, 3),
            'SMILES': LIGAND_SMILES,
            'pro_seq': sq
        })

# Prepare output DataFrame
df_new = pd.DataFrame(results)

# Check if CSV file exists
if os.path.exists(CSV_OUT):
    # Read existing data
    df_existing = pd.read_csv(CSV_OUT)
    # Combine new and existing data
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    # Write the combined data back to CSV
    df_combined.to_csv(CSV_OUT, index=False)
    print(f"Results appended to {CSV_OUT}")
else:
    # If CSV does not exist, write new data
    df_new.to_csv(CSV_OUT, index=False)
    print(f"Results saved to {CSV_OUT}")