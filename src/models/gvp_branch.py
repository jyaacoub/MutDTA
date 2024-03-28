from torch import nn
import torch
from torch_scatter import scatter_mean
from src.models.utils import GVP, GVPConvLayer, LayerNorm

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
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.1):
        
        super(GVPBranchProt, self).__init__()
        
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
            nn.Linear(2*ns, 1)
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
        batch = data.batch if hasattr(data, 'batch') else None #TODO: check if this is correct
        
        if seq is not None:
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
        
        return self.dense(out).squeeze(-1) + 0.5, out, out1, h_V