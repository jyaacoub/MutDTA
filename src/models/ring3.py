import torch
from torch import nn

from torch_geometric import nn as nn_g
from torch_geometric import data as geo_data
from torch_geometric.utils import dropout_edge, dropout_node
from torch_geometric.nn import summary
from torch_geometric.nn import (GCNConv, TransformerConv,
                                global_max_pool, 
                                global_mean_pool)

from transformers import AutoTokenizer, EsmModel
from transformers.utils import logging

from src.models.utils import BaseModel

class Ring3Branch(BaseModel):
    """Model using ring3 features for protein branch with no ESM embeddings"""
    def __init__(self, pro_emb_dim=128, output_dim=250, 
                 dropout=0.2, nheads_pro=5,
                 
                 # Feature input sizes:
                 num_features_pro=54, # esm has 320d embeddings original feats is 54
                 edge_dim_pro=6, # edge dim for protein branch from RING3
                 ):
        
        super(Ring3Branch, self).__init__()
        # PROTEIN BRANCH: (cant use nn_g.Sequential since we want dropout and edge attr)
        self.pro_gnn1 = TransformerConv(num_features_pro, pro_emb_dim, 
                                        edge_dim=edge_dim_pro, heads=nheads_pro,
                                        dropout=dropout)
        self.pro_gnn2 = TransformerConv(pro_emb_dim*nheads_pro, pro_emb_dim*2,
                                        edge_dim=edge_dim_pro, heads=nheads_pro,
                                        dropout=dropout)
        self.pro_gnn3 = TransformerConv(pro_emb_dim*2*nheads_pro, pro_emb_dim*2,
                                        edge_dim=edge_dim_pro, heads=nheads_pro,
                                        concat=False, dropout=dropout)
        
        self.pro_fc = nn.Sequential(
            nn.Linear(pro_emb_dim*2, pro_emb_dim*2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(pro_emb_dim*2, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        # misc functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        ei = data.edge_index
        edge_attr = data.edge_weight
        
        #### Graph NN ####
        target_x = self.relu(data.x)
        
        # GNN layers:
        # NOTE: dropout is done to the attention coefficients in the TransformerConv
        xt = self.pro_gnn1(target_x, ei, edge_attr)
        xt = self.relu(xt)
        xt = self.pro_gnn2(xt, ei, edge_attr)
        xt = self.relu(xt)
        xt = self.pro_gnn3(xt, ei, edge_attr)
        xt = self.relu(xt)

        # flatten/pool
        xt = global_mean_pool(xt, data.batch)  # global avg pooling
        xt = self.relu(xt)
        xt = self.dropout(xt)

        #### FC layers ####
        xt = self.pro_fc(xt)
        return xt
    

class Ring3DTA(BaseModel):
    """Model using ring3 features for protein branch with no ESM embeddings"""
    def __init__(self, pro_emb_dim=128, output_dim=250, 
                 dropout=0.2, dropout_prot=0.2, nheads_pro=5,
                 
                 # Feature input sizes:
                 num_features_mol=78,
                 num_features_pro=54, # esm has 320d embeddings original feats is 54
                 edge_dim_pro=6, # edge dim for protein branch from RING3
                 ):
        
        super(Ring3DTA, self).__init__()

        # LIGAND BRANCH 
        # (NOTE: sequential doesnt work if we want to use dropout or edge weights):
        self.mol_gnn = nn_g.Sequential('x, edge_index, batch', [
            (GCNConv(num_features_mol, num_features_mol*2), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(num_features_mol*2, num_features_mol*4), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(num_features_mol*4, 1024), 'x, edge_index -> x'),
            nn.ReLU(),
            (global_mean_pool, 'x, batch -> x'),
        ])
        
        self.mol_fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # PROTEIN BRANCH:
        self.ring3_pro = Ring3Branch(pro_emb_dim, output_dim, dropout, 
                                     nheads_pro,num_features_pro,
                                     edge_dim_pro)
        
        # CONCATENATION OF BRANCHES:
        self.fc_concat = nn.Sequential(
            nn.Linear(2 * output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )
        
        # misc functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_prot_p = dropout_prot
    
    def forward_pro(self, data):
        return self.ring3_pro(data)
    
    def forward_mol(self, data):
        x = self.mol_gnn(data.x, data.edge_index, data.batch)
        x = self.mol_fc(x)
        return x

    def forward(self, data_pro:geo_data.data.Data, data_mol:geo_data.data.Data):
        """
        Forward pass of the model.

        Parameters
        ----------
        `data_pro` : torch_geometric.data.data.Data
            the protein data
        `data_mol` : torch_geometric.data.data.Data
            the ligand data

        Returns
        -------
        Tensor
            output of the model
        """
        xm = self.forward_mol(data_mol)
        xp = self.forward_pro(data_pro)
        
        # concat
        xc = torch.cat((xm, xp), 1)
        out = self.fc_concat(xc)
        return out
    
class Ring3_ESMDTA(Ring3DTA):
    def __init__(self, esm_head: str = 'facebook/esm2_t6_8M_UR50D', pro_emb_dim=512, output_dim=250, dropout=0.2, dropout_prot=0):
        super().__init__(pro_emb_dim, output_dim, dropout, dropout_prot)
        
        # defining ESM model and tokenizer
        # this will raise a warning since lm head is missing but that is okay since we are not using it:
        logging.set_verbosity(logging.CRITICAL)
        self.esm_tok = AutoTokenizer.from_pretrained(esm_head)
        self.esm_mdl = EsmModel.from_pretrained(esm_head)
        self.esm_mdl.requires_grad_(False) # freeze weights
        
    
    def forward_pro(self, data):
        #### ESM emb ####
        # cls and sep tokens are added to the sequence by the tokenizer
        seq_tok = self.esm_tok(data.pro_seq, 
                               return_tensors='pt', 
                               padding=True) # [B, L_max+2]
        seq_tok['input_ids'] = seq_tok['input_ids'].to(data.x.device)
        seq_tok['attention_mask'] = seq_tok['attention_mask'].to(data.x.device)
        
        esm_emb = self.esm_mdl(**seq_tok).last_hidden_state # [B, L_max+2, emb_dim]
        
        # removing <cls> token
        esm_emb = esm_emb[:,1:,:] # [B, L_max+1, emb_dim]
        
        # removing <sep> token by applying mask
        L_max = esm_emb.shape[1] # L_max+1
        mask = torch.arange(L_max)[None, :] < torch.tensor([len(seq) for seq in data.pro_seq])[:, None]
        mask = mask.flatten(0,1) # [B*L_max+1]
        
        # flatten from [B, L_max+1, emb_dim] 
        esm_emb = esm_emb.flatten(0,1) # to [B*L_max+1, emb_dim]
        esm_emb = esm_emb[mask] # [B*L, emb_dim]
        
        # append esm embeddings to protein input
        target_x = torch.cat((esm_emb, data.x), axis=1)
        #  ->> [B*L, emb_dim+feat_dim]

        ei = data.edge_index
        # if edge_weight doesnt exist no error is thrown it just passes it as None
        ew = data.edge_weight if (self.edge_weight is not None and 
                                  self.edge_weight != 'binary') else None
        
        #### Graph NN ####
        target_x = self.relu(target_x)
        # WARNING: dropout_node doesnt work if `ew` isnt also dropped out
        # ei_drp, _, _ = dropout_node(ei, p=self.dropout_prot_p, num_nodes=target_x.shape[0], 
        #                                 training=self.training)
        # GNN layers:
        xt = self.pro_conv1(target_x, ei, ew)
        xt = self.relu(xt)
        # ei_drp, _, _ = dropout_node(ei, p=self.dropout_prot_p, num_nodes=target_x.shape[0], 
        #                                 training=self.training)
        xt = self.pro_conv2(xt, ei, ew)
        xt = self.relu(xt)
        # ei_drp, _, _ = dropout_node(ei, p=self.dropout_prot_p, num_nodes=target_x.shape[0], 
        #                                 training=self.training)
        xt = self.pro_conv3(xt, ei, ew)
        xt = self.relu(xt)

        # flatten/pool
        xt = global_mean_pool(xt, data.batch)
        xt = self.relu(xt)
        xt = self.dropout(xt)

        #### FC layers ####
        xt = self.pro_fc_g1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
    
        xt = self.pro_fc_g3(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
    
        xt = self.pro_fc_g3(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt
    