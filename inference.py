# %%
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import target_to_graph
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
import logging
from src.utils.loader import Loader
logging.getLogger().setLevel(logging.DEBUG)

m, _ = Loader.load_tuned_model('davis_esm', fold=1)

# %%
m(pro, lig)

# %%
