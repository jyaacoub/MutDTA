import torch
from torch import nn

from torch_geometric.nn import (GCNConv,
                                global_mean_pool as gep)
from torch_geometric.data import Data as Data_g
from torch_geometric import nn as nn_g

from src.models.utils import BaseModel
from src.models.branches import GVPBranchProt, GVPBranchLigand, ESMBranch


from src.models.prior_work import DGraphDTA
from src.models.ring3 import Ring3Branch


class GVPLigand_DGPro(DGraphDTA):
    """
    DG model with GVP Ligand branch
    """
    def __init__(self, num_features_pro=54,
                 num_features_mol=78, 
                 output_dim=512,
                 dropout=0.2,
                 num_GVPLayers=3,
                 edge_weight_opt='binary', **kwargs):
        output_dim = int(output_dim)
        super(GVPLigand_DGPro, self).__init__(num_features_pro, 
                                                num_features_mol, output_dim, 
                                                dropout, edge_weight_opt)
        
        self.gvp_ligand = GVPBranchLigand(num_layers=num_GVPLayers, 
                                          final_out=output_dim,
                                          drop_rate=dropout)
        
        self.dense_out = nn.Sequential(
            nn.Linear(2*output_dim, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(128, 1),        
        )
    
    def forward_mol(self, data):
        return self.gvp_ligand(data)
    
    def forward(self, data_pro, data_mol):
        xm = self.forward_mol(data_mol)
        xp = self.forward_pro(data_pro)

        xc = torch.cat((xm, xp), 1)
        return self.dense_out(xc)
    
class GVPL_ESM(nn.Module):
    def __init__(self, 
                 pro_num_feat=320,pro_emb_dim=512, pro_dropout_gnn=0.0, pro_extra_fc_lyr=False,
                 num_GVPLayers=3,
                 dropout=0.2,
                 output_dim=512,
                 **kwargs):
        output_dim = int(output_dim)
        super(GVPLigand_DGPro, self).__init__()
        
        self.gvp_ligand = GVPBranchLigand(num_layers=num_GVPLayers, 
                                          final_out=output_dim,
                                          drop_rate=dropout)
        
        self.esm_branch = ESMBranch(num_feat=pro_num_feat, emb_dim=pro_emb_dim, 
                                    dropout_gnn=pro_dropout_gnn, 
                                    extra_fc_lyr=pro_extra_fc_lyr,
                                    output_dim=output_dim, dropout=dropout)
        
        self.dense_out = nn.Sequential(
            nn.Linear(2*output_dim, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            
            nn.Linear(128, 1),        
        )
        
    def forward_pro(self, data):
        return self.esm_branch(data)
    
    def forward_mol(self, data):
        return self.gvp_ligand(data)
    
    def forward(self, data_pro, data_mol):
        xm = self.forward_mol(data_mol)
        xp = self.forward_pro(data_pro)

        xc = torch.cat((xm, xp), 1)
        return self.dense_out(xc)


class GVPLigand_RNG3(BaseModel):
    def __init__(self, dropout=0.2, pro_emb_dim=128, output_dim=250, 
                  nheads_pro=5,
                 
                 # Feature input sizes:
                 num_features_pro=54, # esm has 320d embeddings original feats is 54
                 edge_dim_pro=6, # edge dim for protein branch from RING3
                 **kwargs
                 ):
        
        super(GVPLigand_RNG3, self).__init__()

        # LIGAND BRANCH 
        self.forward_mol = GVPBranchLigand(final_out=output_dim, drop_rate=dropout)

        # PROTEIN BRANCH:
        self.forward_pro = Ring3Branch(pro_emb_dim, output_dim, dropout, 
                                     nheads_pro,num_features_pro,
                                     edge_dim_pro)
        
        # CONCATENATION OF BRANCHES:
        self.dense_out = nn.Sequential(
            nn.Linear(2*output_dim, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, 1),        
        )

    def forward(self, data_pro:Data_g, data_mol:Data_g):
        xm = self.forward_mol(data_mol)
        xp = self.forward_pro(data_pro)
        
        # concat
        xc = torch.cat((xm, xp), 1)
        return self.dense_out(xc)
    

class GVPModel(BaseModel):
    def __init__(self, num_features_mol=78,
                 output_dim=128, dropout=0.2,
                 dropout_prot=0.0, **kwargs):
        
        super(GVPModel, self).__init__()

        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        # Protein graph:
        self.pro_branch = GVPBranchProt(node_in_dim=(6, 3), node_h_dim=(6, 3),
                      edge_in_dim=(32, 1), edge_h_dim=(32, 1),
                      seq_in=False, num_layers=3,
                      drop_rate=dropout_prot,
                      final_out=output_dim)
        
        ## OUTPUT LAYERS:
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1) # 1 output (binding affinity)            
        
    def forward_mol(self, data):
        x = self.mol_conv1(data.x, data.edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, data.edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, data.edge_index)
        x = self.relu(x)
        x = gep(x, data.batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.relu(self.mol_fc_g2(x))
        x = self.dropout(x)
        return x

    def forward(self, data_pro, data_mol):
        """
        Forward pass of the model.

        Parameters
        ----------
        `data_pro` : _type_
            the protein data
        `data_mol` : _type_
            the ligand data

        Returns
        -------
        _type_
            output of the model
        """
        xm = self.forward_mol(data_mol)
        xp = self.pro_branch(data_pro)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((xm, xp), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
