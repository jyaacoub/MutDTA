# %%
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import target_to_graph
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights
from src import cfg
import torch
import torch_geometric as torchg
import numpy as np

DATA = cfg.DATA_OPT.davis
lig_feature = cfg.LIG_FEAT_OPT.original
lig_edge = cfg.LIG_EDGE_OPT.binary
pro_feature = cfg.PRO_FEAT_OPT.nomsa
pro_edge = cfg.PRO_EDGE_OPT.binary

lig_seq = "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4"

#%% build ligand graph
mol_feat, mol_edge = smile_to_graph(lig_seq, lig_feature=lig_feature, lig_edge=lig_edge)
lig = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge),
                    lig_seq=lig_seq)

#%% build protein graph
# predicted using - https://zhanggroup.org/NeBcon/
prot_id = 'EGFR(L858R)'
pro_seq = 'MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA'
cmap_p = f'/cluster/home/t122995uhn/projects/data/davis/pconsc4/{prot_id}.npy'

pro_cmap = np.load(cmap_p)
# updated_seq is for updated foldseek 3di combined seq
updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pro_seq, 
                                                    contact_map=pro_cmap,
                                                    threshold=8.0 if DATA is cfg.DATA_OPT.PDBbind else -0.5, 
                                                    pro_feat=pro_feature)
pro_feat = torch.Tensor(extra_feat)



pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                    edge_index=torch.LongTensor(edge_idx),
                    pro_seq=updated_seq, # Protein sequence for downstream esm model
                    prot_id=prot_id,
                    edge_weight=None)

#%% Loading the model
from src.utils.loader import Loader
from src import TUNED_MODEL_CONFIGS
import os

def reformat_kwargs(model_kwargs):
    return {
        'model': model_kwargs['model'],
        'data': model_kwargs['dataset'],
        'pro_feature': model_kwargs['feature_opt'],
        'edge': model_kwargs['edge_opt'],
        'batch_size': model_kwargs['batch_size'],
        'lr': model_kwargs['lr'],
        'dropout': model_kwargs['architecture_kwargs']['dropout'],
        'n_epochs': model_kwargs.get('n_epochs', 2000),  # Assuming a default value for n_epochs
        'pro_overlap': model_kwargs.get('pro_overlap', False),  # Assuming a default or None
        'fold': model_kwargs.get('fold', 0),  # Assuming a default or None
        'ligand_feature': model_kwargs['lig_feat_opt'],
        'ligand_edge': model_kwargs['lig_edge_opt']
    }


model_kwargs = reformat_kwargs(TUNED_MODEL_CONFIGS['davis_esm'])

MODEL_KEY = Loader.get_model_key(**model_kwargs)

model_p_tmp = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model_tmp'
model_p = f'{cfg.MODEL_SAVE_DIR}/{MODEL_KEY}.model'

# MODEL_KEY = 'DDP-' + MODEL_KEY # distributed model
model_p = model_p if os.path.isfile(model_p) else model_p_tmp
assert os.path.isfile(model_p), f"MISSING MODEL CHECKPOINT {model_p}"

print(model_p)
# %%
args = model_kwargs
model = Loader.init_model(model=args['model'], pro_feature=args['pro_feature'], 
                          pro_edge=args['edge'], **args['architecture_kwargs'])


