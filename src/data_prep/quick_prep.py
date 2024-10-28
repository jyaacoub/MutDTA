import os

import torch
import torch_geometric as torchg

from src import cfg
from src.utils.residue import Chain
from src.data_prep.feature_extraction.ligand import smile_to_graph
from src.data_prep.feature_extraction.protein import target_to_graph
from src.data_prep.feature_extraction.protein_edges import get_target_edge_weights


# Get initial pkd value:
def get_protein_features(pdb_file_path, feature_opt, edge_opt, prot_id=None, cmap_thresh=8.0):
    prot_id = prot_id or os.path.basename(pdb_file_path).split('.pdb')[0]
    pdb = Chain(pdb_file_path)
    pro_cmap = pdb.get_contact_map()

    updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pdb.sequence, 
                                                        contact_map=pro_cmap,
                                                        threshold=cmap_thresh, 
                                                        pro_feat=feature_opt)
    pro_edge_weight = None
    if edge_opt != cfg.PRO_EDGE_OPT.binary:
        # includes edge_attr like ring3
        pro_edge_weight = get_target_edge_weights(pdb_file_path, pdb.sequence, 
                                            edge_opt=edge_opt,
                                            cmap=pro_cmap,
                                            af_confs=pdb_file_path,
                                            n_modes=5, n_cpu=4)
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
    return pro, pdb


def get_ligand_features(lig_smile, lig_feat, lig_edge, lig_sdf=None):
    if lig_feat == cfg.LIG_FEAT_OPT.gvp:
        from src.data_prep.feature_extraction.gvp_feats import GVPFeaturesLigand
        return GVPFeaturesLigand().featurize_as_graph(lig_sdf) # returns torch_geometric.data.Data(x=coords, edge_index=edge_index, name=name,node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)
    else:
        mol_feat, mol_edge = smile_to_graph(lig_smile, lig_feature=lig_feat, lig_edge=lig_edge)
        return torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge), lig_seq=lig_smile)