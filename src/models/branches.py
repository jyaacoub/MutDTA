import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_mean_pool as gep
from torch_geometric.utils import dropout_edge, dropout_node
from torch_geometric import data as geo_data
from torch_geometric.nn import summary

from transformers import AutoTokenizer, EsmModel
from transformers.utils import logging

from torch_scatter import scatter_mean 
from src.models.utils import GVP, GVPConvLayer, LayerNorm

class ESMBranch(nn.Module):
    def __init__(self, esm_head:str='facebook/esm2_t6_8M_UR50D', 
                 num_feat=320, emb_dim=512, output_dim=128, dropout=0.2, 
                 dropout_gnn=0.0, extra_fc_lyr=False, esm_only=True,
                 edge_weight='binary'):
        
        super(ESMBranch, self).__init__()
        
        self.esm_only = esm_only
        self.edge_weight = edge_weight

        # Protein graph:
        self.conv1 = GCNConv(num_feat, emb_dim)
        self.conv2 = GCNConv(emb_dim, emb_dim * 2)
        self.conv3 = GCNConv(emb_dim * 2, emb_dim * 4)
        
        if not extra_fc_lyr:
            self.fc_g1 = nn.Linear(emb_dim * 4, 1024)
        else:
            self.fc_g1 = nn.Linear(emb_dim * 4, emb_dim * 2)
            self.fc_g1b = nn.Linear(emb_dim * 2, 1024)
            
        self.extra_fc_lyr = extra_fc_lyr
        self.fc_g2 = nn.Linear(1024, output_dim)
            
        # this will raise a warning since lm head is missing but that is okay since we are not using it:
        # prev_v = logging.get_verbosity()
        logging.set_verbosity(logging.CRITICAL)
        self.esm_tok = AutoTokenizer.from_pretrained(esm_head)
        self.esm_mdl = EsmModel.from_pretrained(esm_head)
        self.esm_mdl.requires_grad_(False) # freeze weights

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # note that dropout for edge and nodes is handled by torch_geometric in forward pass
        self.dropout_gnn = dropout_gnn
    
    def forward(self, data):
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
        # if edge_weight doesnt exist no error is thrown it just passes it as None
        ew = data.edge_weight if (self.edge_weight is not None and 
                                  self.edge_weight != 'binary') else None
        
        target_x = self.relu(target_x)
        ei_drp, _, _ = dropout_node(ei, p=self.dropout_gnn, num_nodes=target_x.shape[0], 
                                        training=self.training)
        
        # conv1
        xt = self.conv1(target_x, ei_drp, ew)
        xt = self.relu(xt)
        ei_drp, _, _ = dropout_node(ei, p=self.dropout_gnn, num_nodes=target_x.shape[0], 
                                        training=self.training)
        # conv2
        xt = self.conv2(xt, ei_drp, ew)
        xt = self.relu(xt)
        ei_drp, _, _ = dropout_node(ei, p=self.dropout_gnn, num_nodes=target_x.shape[0], 
                                        training=self.training)
        # conv3
        xt = self.conv3(xt, ei_drp, ew)
        xt = self.relu(xt)

        # flatten/pool
        xt = gep(xt, data.batch)  # global pooling
        xt = self.relu(xt)
        xt = self.dropout(xt)

        #### FC layers ####
        xt = self.fc_g1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        
        if self.extra_fc_lyr:
            xt = self.fc_g1b(xt)
            xt = self.relu(xt)
            xt = self.dropout(xt)
        
        xt = self.fc_g2(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        return xt


# Adapted from https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/models.py 
class GVPBranchProt(nn.Module):
    '''
    GVP model for protein branch.
    
    This follows the same architecture as the Model Quality Assessment model
    from the manuscript (https://github.com/drorlab/gvp-pytorch/blob/main/gvp)
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param edge_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim=(6, 3), node_h_dim=(6, 3), 
                edge_in_dim=(32, 1), edge_h_dim=(32, 1), final_out=1,
                seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(GVPBranchProt, self).__init__()
        
        self.seq = seq_in
        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        # combines scalar and vector features into a single output [N, 6]
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))
            
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, final_out)
        )

    def forward(self, data):      
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        h_V, h_E = (data.node_s, data.node_v), (data.edge_s, data.edge_v)
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        if self.seq:
            seq = data.seq
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out1 = self.W_out(h_V)
        
        # aggregate node embeddings using global mean pooling:
        if batch is None: out = out1.mean(dim=0, keepdims=True)
        else: out = scatter_mean(out1, batch, dim=0)
        
        return self.dense(out).squeeze(-1) + 0.5
    
    

class GVPBranchLigand(nn.Module):
    """
    Adapted from https://github.com/luoyunan/KDBNet/blob/main/kdbnet/model.py
    """
    
    def __init__(self, 
        node_in_dim=[66, 1], node_h_dim=[128, 64],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1,
        final_out=128
    ):
        """
        Node features are 
            - 66 "scalar feat" (see GVPFeaturesLigand._build_atom_feature)
                - One hot vectors... for symbol(44), degree(7), totalHs(7), 
                  valence count(7), Aromaticity(1)
            - 1 "vector feat" for coordinate positions for atoms

            
        edge features are:
            - 16 "scalar feats" 
                - 16d RBF embedding of the edge vectors representing the distance between them
            - 1 "vector feat" representing the normalized distances between edges
        
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(GVPBranchLigand, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        # same as protein branch, here we ignore the "vector features" and only return 
        # scalar outputs from nodes.
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
            )
        
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, final_out)
        )

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # per-graph mean
        out = torch_geometric.nn.global_add_pool(out, batch)

        return self.dense(out)