from typing import Any, Mapping
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (GCNConv, GATConv, 
                                global_max_pool as gmp, 
                                global_mean_pool as gep)
from torch_geometric.utils import dropout_adj

from torch_geometric.nn import summary
from torch_geometric import data as geo_data

from src.models.utils import BaseModel

from transformers import AutoTokenizer, EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

class EsmDTA(BaseModel):
    def __init__(self, esm_head:str='facebook/esm2_t6_8M_UR50D', 
                 num_features_pro=320, num_features_mol=78, 
                 output_dim=128, dropout=0.2):
        super(EsmDTA, self).__init__()

        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        emb_feat= 54 # to ensure constant embedding size regardless of input size (for fair comparison)
        self.pro_conv1 = GCNConv(num_features_pro, emb_feat)
        self.pro_conv2 = GCNConv(emb_feat, emb_feat * 2)
        self.pro_conv3 = GCNConv(emb_feat * 2, emb_feat * 4)
        self.pro_fc_g1 = nn.Linear(emb_feat * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        
        # this will raise a warning since lm head is missing but that is okay since we are not using it:
        self.esm_tok = AutoTokenizer.from_pretrained(esm_head)
        self.esm_mdl = EsmModel.from_pretrained(esm_head)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1) # 1 output (binding affinity)
    
    def forward_pro(self, data):        
        # TODO: might need to reshape??
        
        # leaving cls and sep tokens in the sequence
        seq_tok = self.esm_tok(data.seq, return_tensors='pt', padding=True)
        
        esm_emb = self.esm_mdl(seq_tok['input_ids'], seq_tok['attention_mask']).last_hidden_state
        
        # append esm embeddings to protein input
        target_x = torch.cat((esm_emb, data.x), axis=2) 
        #  ->> [batch, seq_len, emb_dim+feat_dim]

        xt = self.pro_conv1(target_x, data.edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, data.edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, data.edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, data.batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        return xt
    
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
        x = self.mol_fc_g2(x)
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
        xp = self.forward_pro(data_pro)

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
    
    def __str__(self) -> str:
        main_str = super().__str__()
        
        prot_shape = (self.pro_conv1.in_channels, self.pro_conv1.out_channels)
        lig_shape = (self.mol_conv1.in_channels, self.mol_conv1.out_channels)
        device = next(self.parameters()).device
        
        prot = geo_data.Data(x=torch.Tensor(*prot_shape), # node feature matrix
                            edge_index=torch.LongTensor([[0,1]]).transpose(1, 0),
                            y=torch.FloatTensor([1])).to(device)
        lig = geo_data.Data(x=torch.Tensor(*lig_shape), # node feature matrix
                            edge_index=torch.LongTensor([[0,1]]).transpose(1, 0),
                            y=torch.FloatTensor([1])).to(device)
        
        model_summary = summary(self, lig, prot)

        return main_str + '\n\n' + model_summary