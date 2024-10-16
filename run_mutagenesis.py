import argparse
parser = argparse.ArgumentParser(description='Runs Mutagenesis on an input PDB file and a given ligand SMILES.')
parser.add_argument('--ligand_smile', type=str, required=True, help='Ligand SMILES string.')
parser.add_argument('--ligand_smile_name', type=str, required=True, help='Ligand SMILES name, required for output path.')
parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file.')
parser.add_argument('--out_path', type=str, default='./', 
                    help='Output directory path to save resulting mutagenesis numpy matrix with predicted pkd values')
parser.add_argument('--res_start', type=int, default=0, help='Start index for mutagenesis (zero-indexed).')
parser.add_argument('--res_end', type=int, default=float('inf'), help='End index for mutagenesis.')

parser.add_argument('--model_opt', type=str, default='davis_DG', 
                    choices=['davis_DG',    'davis_gvpl',   'davis_esm', 
                             'kiba_DG',     'kiba_esm',     'kiba_gvpl',
                             'PDBbind_DG',  'PDBbind_esm',  'PDBbind_gvpl'],
                    help='Model option. See MutDTA/src/__init__.py for details.')
parser.add_argument('--fold', type=int, default=1, 
                    help='Which model fold to use (there are 5 models for each option due to 5-fold CV).')

args = parser.parse_args()

# Assign variables
LIGAND_SMILE = args.ligand_smile
LIGAND_SMILE_NAME = args.ligand_smile_name
PDB_FILE = args.pdb_file
OUT_PATH = args.out_path
MODEL_OPT = args.model_opt
FOLD = args.fold
RES_START = args.res_start
RES_END = args.res_end

import logging
logging.getLogger().setLevel(logging.DEBUG)
logging.debug("#"*50)
logging.debug(f"LIGAND_SMILE: {LIGAND_SMILE}")
logging.debug(f"LIGAND_SMILE_NAME: {LIGAND_SMILE_NAME}")
logging.debug(f"PDB_FILE: {PDB_FILE}")
logging.debug(f"OUT_PATH: {OUT_PATH}")
logging.debug(f"MODEL_OPT: {MODEL_OPT}")
logging.debug(f"FOLD: {FOLD}")
logging.debug(f"RES_START: {RES_START}")
logging.debug(f"RES_END: {RES_END}")
logging.debug("#"*50)

import os
import numpy as np
import torch
import torch_geometric as torchg
from tqdm import tqdm

from src import cfg
from src import TUNED_MODEL_CONFIGS

from src.utils.loader import Loader
from src.utils.residue import ResInfo, Chain
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import target_to_graph
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights
from src.utils.residue import ResInfo, Chain

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = TUNED_MODEL_CONFIGS[MODEL_OPT]
PDB_FILE_NAME = os.path.basename(PDB_FILE).split('.pdb')[0]

# Get initial pkd value:
def get_protein_features(pdb_file_path, cmap_thresh=8.0):
    pdb = Chain(pdb_file_path)
    pro_cmap = pdb.get_contact_map()

    updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pdb.sequence, 
                                                        contact_map=pro_cmap,
                                                        threshold=cmap_thresh, 
                                                        pro_feat=MODEL_PARAMS['feature_opt'])
    pro_edge_weight = None
    if MODEL_PARAMS['edge_opt'] in cfg.OPT_REQUIRES_CONF:
        raise NotImplementedError(f"{MODEL_PARAMS['edge_opt']} is not supported since it requires "+\
                                    "multiple conformation files to run and generate edges.")
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
                            prot_id=PDB_FILE_NAME,
                            edge_weight=pro_edge_weight)
    return pro, pdb

##################################################
### Loading the model and get original pkd value #
##################################################
m, _ = Loader.load_tuned_model(MODEL_OPT, fold=FOLD)
m.to(DEVICE)
m.eval()

# build ligand graph
mol_feat, mol_edge = smile_to_graph(LIGAND_SMILE, lig_feature=MODEL_PARAMS['lig_feat_opt'], lig_edge=MODEL_PARAMS['lig_edge_opt'])
lig = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge), lig_seq=LIGAND_SMILE)

# build protein graph
pro, pdb_original = get_protein_features(PDB_FILE)
original_seq = pdb_original.sequence

original_pkd = m(pro.to(DEVICE), lig.to(DEVICE))
print("Original pkd:", original_pkd)


##################################################
### Mutate and regenerate graphs #################
##################################################
# zero indexed res range to mutate:
res_range = (max(RES_START, 0),
            min(RES_END, len(original_seq)))

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
            
            muta[i,j] = m(pro.to(DEVICE), lig.to(DEVICE))
            
            # delete after use
            os.remove(out_pdb_fp)


# Save mutagenesis matrix
OUT_DIR = f'{OUT_PATH}/{LIGAND_SMILE_NAME}/{MODEL_OPT}'
os.makedirs(OUT_DIR)
OUT_FP = f"{OUT_DIR}/{res_range[0]}_{res_range[1]}.npy"
print("Saving mutagenesis numpy matrix to", OUT_FP)
np.save(OUT_FP, muta)