import torch
from torch import nn

from torch_geometric.nn import (GCNConv,
                                global_mean_pool as gep)

from src.models.utils import BaseModel
from src.models.gvp_branch import GVPBranchProt

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