from typing import Any, Mapping
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_geometric.nn import (GCNConv, GATConv, 
                                global_max_pool as gmp, 
                                global_mean_pool as gep)
from torch_geometric.utils import dropout_adj

from torch_geometric.nn import summary
from torch_geometric import data as geo_data

from transformers import AutoTokenizer, EsmModel
from transformers.utils import logging

from src.utils.loader import Loader
from src.models.utils import BaseModel

class EsmAttentionDTA(BaseModel):
    """
    Non-Graph attention-based model. This model only uses ESM embeddings as the protein features.
    """
    def __init__(self, esm_head:str='facebook/esm2_t6_8M_UR50D', 
                 num_features_pro=320, 
                 num_features_mol=78, nhead_mol=8,
                 output_dim=320, dropout=0.2):
        super(EsmAttentionDTA, self).__init__(pro_feat=None, edge_weight_opt=None)

        # Mol graph:
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # protein encoding:
        # this will raise a warning since lm head is missing but that is okay since we are not using it:
        logging.set_verbosity(logging.CRITICAL)
        self.esm_tok = AutoTokenizer.from_pretrained(esm_head)
        self.esm_mdl = EsmModel.from_pretrained(esm_head)
        self.esm_mdl.requires_grad_(False) # freeze weights
        
        self.pro_encode = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features_pro, nhead=nhead_mol*2,
                                       dim_feedforward=num_features_pro*4, dropout=0.2),
            num_layers=4
        )
        
        # final output
        self.fc_out = nn.Sequential(
            nn.Linear(output_dim*2, output_dim*4), #increase dim to minimize info loss due to ReLU
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim*4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1) # 1 output (binding affinity)
        )
        
    def forward_pro(self, data):
        # cls and sep tokens are added to the sequence by the tokenizer
        seq_tok = self.esm_tok(data.pro_seq, 
                               return_tensors='pt', 
                               padding=True) # [B, L_max+2]
        seq_tok['input_ids'] = seq_tok['input_ids'].to(data.x.device)
        seq_tok['attention_mask'] = seq_tok['attention_mask'].to(data.x.device)
        
        esm_emb = self.esm_mdl(**seq_tok).last_hidden_state # [B, L_max+2, emb_dim]
        x = self.pro_encode(esm_emb, 
                            src_key_padding_mask=~seq_tok['attention_mask'].bool().T)
        
        # [B, L_max+2, emb_dim]
        # pool data -> [B, emb_dim]
        x = torch.mean(x, dim=1)
        return x
        
    def forward_mol(self, data):
        x = self.mol_conv1(data.x, data.edge_index)
        x = self.relu(x)

        x = self.mol_conv2(x, data.edge_index)
        x = self.relu(x)

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
        xm = self.forward_mol(data_mol)
        xp = self.forward_pro(data_pro)

        # concat
        xc = torch.cat((xm, xp), 1)
        
        return self.fc_out(xc)
        

class EsmDTA(BaseModel):
    def __init__(self, esm_head:str='facebook/esm2_t6_8M_UR50D', 
                 num_features_pro=320, pro_emb_dim=54, num_features_mol=78, 
                 output_dim=128, dropout=0.2, pro_feat='esm_only', edge_weight_opt='binary'):
        
        super(EsmDTA, self).__init__(pro_feat, edge_weight_opt)

        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, pro_emb_dim)
        self.pro_conv2 = GCNConv(pro_emb_dim, pro_emb_dim * 2)
        self.pro_conv3 = GCNConv(pro_emb_dim * 2, pro_emb_dim * 4)
        self.pro_fc_g1 = nn.Linear(pro_emb_dim * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        
        # this will raise a warning since lm head is missing but that is okay since we are not using it:
        logging.set_verbosity(logging.CRITICAL)
        self.esm_tok = AutoTokenizer.from_pretrained(esm_head)
        self.esm_mdl = EsmModel.from_pretrained(esm_head)
        self.esm_mdl.requires_grad_(False) # freeze weights

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1) # 1 output (binding affinity)            
    
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
        
        if self.esm_only:
            target_x = esm_emb # [B*L, emb_dim]
        else:
            # append esm embeddings to protein input
            target_x = torch.cat((esm_emb, data.x), axis=1)
            #  ->> [B*L, emb_dim+feat_dim]

        #### Graph NN ####
        ei = data.edge_index
        ew = data.edge_weight if self.edge_weight else None

        # if edge_weight doesnt exist no error is thrown it just passes it as None
        xt = self.pro_conv1(target_x, ei, ew)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, ei, ew)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, ei, ew)
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