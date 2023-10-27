import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = '../hf_models/'

from torch import nn
from torch_geometric.nn import (GCNConv, global_mean_pool as gep)

from transformers import AutoTokenizer, AutoModel

from src.models.prior_work import DGraphDTA

class DGraphDTALigand(DGraphDTA):
    def __init__(self, ligand_feature='original', ligand_edge='binary', output_dim=128, *args, **kwargs):
        super(DGraphDTA, self).__init__(*args, **kwargs)

        print('DGraphDTA Loaded')
        num_features_mol = 78
        
        #### ChemGPT ####

        tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
        model = AutoModel.from_pretrained("ncfrey/ChemGPT-4.7M")

        # if ligand_feature == 'some new feature list':
        #       num_features_mol = updated number
        
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
    
    def forward_mol(self, data_mol):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x