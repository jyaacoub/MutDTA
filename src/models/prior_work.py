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

################ DGraphDTA: ################
class DGraphDTA(BaseModel):
    """
    ORIGINAL model
    Improves upon GraphDTA by providing contact maps for a better representation of the protein 
    (instead of just a convolution like in DeepDTA)
        See: https://github.com/595693085/DGraphDTA
        paper: https://pubs.rsc.org/en/content/articlelanding/2020/ra/d0ra02297g
    """
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(DGraphDTA, self).__init__()

        print('DGraphDTA Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
    
    def forward_pro(self, data_pro):
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        return xt
    
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
    


class DGraphDTAImproved(DGraphDTA):
    """
    My version of DGraphDTA
    
    +Shannon
    +conv layer
    +double size of output dim
    """
    def __init__(self, n_output=1, 
                 num_features_pro=34, 
                 # changed from 54-21 = 33 + 1 = 34 due to use of shannon entropy 
                 # instead of full 21XL PSSM matrix
                 num_features_mol=78, 
                 output_dim=512, # increased to 512
                 dropout=0.2):
        super(DGraphDTAImproved, self).__init__(n_output, num_features_pro, 
                                                num_features_mol, output_dim, 
                                                dropout)
        
        prev_feat = 54 # previous size was 54
        self.pro_conv1 = GCNConv(num_features_pro, prev_feat)
        self.pro_conv2 = GCNConv(prev_feat, prev_feat * 2)
        self.pro_conv3 = GCNConv(prev_feat * 2, prev_feat * 4)
        self.pro_conv4 = GCNConv(prev_feat * 4, prev_feat * 8)
        
        self.pro_fc_g0 = nn.Linear(prev_feat * 8, prev_feat*4)
        self.pro_fc_g1 = nn.Linear(prev_feat * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)    
    
    def forward_pro(self, data_pro):
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_conv4(xt, target_edge_index)
        xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(xt)
        xt = self.pro_fc_g0(xt)
        xt = self.dropout(xt)
        xt = self.pro_fc_g1(xt)
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)
        return xt
    

############ GraphDTA ############
# GCN based model
class GraphDTA(BaseModel):
    """
    Added a graph representation of the ligands to the DeepDTA model (still uses 1d conv for protein)
        See: https://github.com/thinng/GraphDTA
        paper: https://doi.org/10.1093/bioinformatics/btaa921
    """
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):
        super(GraphDTA, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8) #WARNING: they are missing 2 more conv layers
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ### get SMILES input
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        #### get protein input
        target = data.target
        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        #### concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
    
    
if __name__ == "__main__":
    data = 'kiba'
    model = DGraphDTA()
    
    model_file_name = f'results/model_checkpoints/{model._get_name}_{data}_t2.model'
    cuda_name = 'cuda:0'
    
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('cuda_name:', cuda_name)
    print('device:', device)

    
    # loading checkpoint
    cp = torch.load(model_file_name, map_location=device) 
    
    model.load_state_dict(cp)
