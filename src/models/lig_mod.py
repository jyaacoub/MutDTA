import torch
from torch import nn
from torch_geometric.nn import (GCNConv, global_mean_pool as gep)

from transformers import AutoTokenizer, AutoModel
from selfies import encoder

from src.models.prior_work import DGraphDTA

class DGraphDTALigand(DGraphDTA):
    def __init__(self, ligand_feature='original', ligand_edge='binary', output_dim=128, *args, **kwargs):
        super(DGraphDTA, self).__init__(pro_feat=None, edge_weight_opt='binary', *args, **kwargs)

        print('DGraphDTA Loaded')
        num_features_mol = 128
        
        #### ChemGPT ####

        # get tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("../hf_models/models--ncfrey--ChemGPT-4.7M/snapshots/7438a282460b3038e17a27e25b85b1376e9a23e2/", local_files_only=True)
        self.model = AutoModel.from_pretrained("../hf_models/models--ncfrey--ChemGPT-4.7M/snapshots/7438a282460b3038e17a27e25b85b1376e9a23e2/", local_files_only=True)

        self.model.requires_grad_(False) # freeze weights

        # adding a new token '[PAD]' to the tokenizer, and then using it as the padding token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.relu = nn.ReLU()

        # if ligand_feature == 'some new feature list':
        #       num_features_mol = updated number
        

        self.mol_fc_g1 = nn.Linear(num_features_mol, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)
    
    def forward_mol(self, data_mol):
        # get smiles list input
        mol_x = data_mol.lig_seq

        print(data_mol.x.device, self.model.device)

        try:
            # get selifes from smile
            selfies = [encoder(s) for s in mol_x]
        except AttributeError as e:
            print(mol_x)
            raise e


        # get tokens
        res = self.tokenizer(selfies, return_tensors="pt", padding=True)

        res['input_ids'] = res['input_ids'].to(data_mol.x.device)
        res['attention_mask'] = res['attention_mask'].to(data_mol.x.device)
        res['token_type_ids'] = res['token_type_ids'].to(data_mol.x.device)

        # model
        model_output = self.model(**res).last_hidden_state

        # flatten to [L, 128]
        x = torch.mean(model_output, dim=1)

        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)
        return x