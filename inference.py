import os, logging, argparse
parser = argparse.ArgumentParser(description='Runs inference on a given ligand SMILES and an input of pdb files.')
parser.add_argument('-ls','--ligand_smile', type=str, required=True, help='Ligand SMILES string.')
parser.add_argument('-pdb','--pdb_files', type=str, nargs='+', required=True, 
                    help='Paths to the PDB files. This can be the alphaflow output files with the first model for edges. '+\
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
args = parser.parse_args()

# Assign variables
LIGAND_SMILES = args.ligand_smile
LIGAND_ID = args.ligand_id
PDB_FILES = args.pdb_files
CSV_OUT = args.csv_out
MODEL_OPT = args.model_opt
FOLD = args.fold
BATCH_SIZE = args.batch_size

logging.getLogger().setLevel(logging.DEBUG)
logging.info("#"*50)
logging.info(f"LIGAND_SMILE: {LIGAND_SMILES}")
logging.info(f"LIGAND_ID: {LIGAND_ID}")
logging.info(f"PDB_FILES: {PDB_FILES}")
logging.info(f"OUT_PATH: {CSV_OUT}")
logging.info(f"MODEL_OPT: {MODEL_OPT}")
logging.info(f"FOLD: {FOLD}")
logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
logging.info("#"*50)

import os
import numpy as np
import pandas as pd
import torch
import torch_geometric as torchg
from torch_geometric.data import Batch
from tqdm import tqdm

from src import cfg
from src import TUNED_MODEL_CONFIGS

from src.utils.loader import Loader
from src.utils.residue import Chain
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import target_to_graph
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = TUNED_MODEL_CONFIGS[MODEL_OPT]

# Get initial pkd value:
def get_protein_features(pdb_file_path, prot_id=None, cmap_thresh=8.0):
    prot_id = prot_id or os.path.basename(pdb_file_path).split('.pdb')[0]
    pdb = Chain(pdb_file_path)
    pro_cmap = pdb.get_contact_map()

    updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pdb.sequence, 
                                                        contact_map=pro_cmap,
                                                        threshold=cmap_thresh, 
                                                        pro_feat=MODEL_PARAMS['feature_opt'])
    pro_edge_weight = None
    if MODEL_PARAMS['edge_opt'] in cfg.OPT_REQUIRES_CONF:        
        pro_edge_weight = get_target_edge_weights(pdb_file_path, pdb.sequence, 
                                            edge_opt=MODEL_PARAMS['edge_opt'],
                                            cmap=pro_cmap,
                                            af_confs=pdb_file_path,
                                            n_modes=5, n_cpu=4)
    else:
        # includes edge_attr like ring3
        pro_edge_weight = get_target_edge_weights(pdb_file_path, pdb.sequence, 
                                            edge_opt=MODEL_PARAMS['edge_opt'],
                                            cmap=pro_cmap,
                                            n_modes=5, n_cpu=4)
        if pro_edge_weight:
            if len(pro_edge_weight.shape) == 2:
                pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1]])
            elif len(pro_edge_weight.shape) == 3: # has edge attr! (This is our GVPL features)
                pro_edge_weight = torch.Tensor(pro_edge_weight[edge_idx[0], edge_idx[1], :])
    
    pro_feat = torch.Tensor(extra_feat)

    pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                            edge_index=torch.LongTensor(edge_idx),
                            pro_seq=updated_seq, # Protein sequence for downstream esm model
                            prot_id=prot_id,
                            edge_weight=pro_edge_weight)
    return pro

##################################################
### Loading the model                          ###
##################################################
m, _ = Loader.load_tuned_model(MODEL_OPT, fold=FOLD, device=DEVICE)
m.eval()

# Build ligand graph
mol_feat, mol_edge = smile_to_graph(LIGAND_SMILES, lig_feature=MODEL_PARAMS['lig_feat_opt'], lig_edge=MODEL_PARAMS['lig_edge_opt'])
lig_data = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge), lig_seq=LIGAND_SMILES)

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

    print(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_pdb_files)} PDB files.")

    # Build protein graphs for the current batch
    pro_list = []
    for PDB_FILE in batch_pdb_files:
        pro = get_protein_features(PDB_FILE)
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
        predicted_pkd = m(pro_batch, lig_batch)

    # Collect results
    predicted_pkd = predicted_pkd.cpu().numpy().flatten()
    time_stamp = pd.Timestamp("now")
    for pdb_file_name, pkd_value, sq in zip(batch_pdb_file_names, predicted_pkd, pro_batch.pro_seq):
        results.append({
            'TIMESTAMP': time_stamp,
            'model': MODEL_OPT,
            'pdb_file': pdb_file_name,
            'ligand_id': LIGAND_ID,
            'pred_pkd': pkd_value,
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