import math
import numpy as np

import torch
import torch_cluster
import torch.nn.functional as F

import torch_geometric


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class GVPFeatures:
    '''
    Process protein data into graph-based features for GVP model following the original
    implementation in the manuscript (https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py).
    
    Features are mentioned in section 3.2 of the manuscript (pg5 in https://openreview.net/pdf?id=1YLJDvSx6J4).
    
    Protein node embedding h_v(i) (features 1-3 contain orientation information):
        1- "scalar" features: dihedral angles {sine, cosine}o(phi, psi, omega)
       *2- "forward and reverse unit vectors" for each residue
       *3- Unit vector in the imputed sidechain direction
        4- one-hot encoding of amino acid identity
    
    NOTE: no thresholding is done to get the edges, instead we just get top_k neighbors.
    Protein edges for neighbor j of residue i:
       *1- Unit vector in the direction of Ca(j) -> Ca(i)
        2- encoding of the distance between Ca(i) and Ca(j) in terms of a Gausian RBF
        3- positional embeddings for the edge (mimics positional encoding of original transformers)
        
    * indicates "vector" features, all others are "scalar" features.
    '''
    def __init__(self, num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu"):
        
        super(GVPFeatures, self).__init__()
        
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        
        # amino acid to number mapping
        # QUESTION: why is this ordered like this??
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
    
    def featurize_as_graph(self, prot_id, prot_coords, prot_seq) -> torch_geometric.data.Data:
        """
        Converts inputs into a graph data object for GVP model.
        
        prot_coords: For each structure, coords should be a 
                [num_residues x 4 x 3] nested list of the positions of the 
                backbone C-alpha, N, and C atoms of each residue (in that order).
                    - {'CA', 'N', 'C'}
        
        """
        with torch.no_grad():
            coords = torch.as_tensor(prot_coords, 
                                     device=self.device, dtype=torch.float32)   
            seq = torch.as_tensor([self.letter_to_num[a] for a in prot_seq],
                                  device=self.device, dtype=torch.long)
            
            mask = torch.isfinite(coords.sum(dim=(1,2))) # remove nan values
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            #NOTE: no thresholding is done for getting edges, instead we just get top_k neighbors
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k) 
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] # unit vector in the direction of Ca(j) -> Ca(i)
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)     # 6 scalar features (sine and cosine of phi, psi, omega)
            orientations = self._orientations(X_ca) # 2 vector features (forward and reverse unit vectors)
            sidechains = self._sidechains(coords)   # 1 vector feature (unit vector in the imputed sidechain direction)
            
            # scalar features:
            node_s = dihedrals
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            
            # vector features:
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            # remove nan values
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=prot_id,
                                         node_s=node_s, node_v=node_v,
                                         
                                         edge_s=edge_s, edge_v=edge_v,
                                         
                                         edge_index=edge_index, mask=mask)
        return data
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 