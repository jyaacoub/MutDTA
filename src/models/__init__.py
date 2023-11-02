from src.models.prior_work import DGraphDTA, GraphDTA
from src.models.pro_mod import EsmDTA
from src.models.utils import BaseModel

def display_models():
    import torch
    from torch_geometric.nn import summary
    from torch_geometric import data as geo_data

    cuda_name = 'cuda:0'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('cuda_name:', cuda_name)
    print('device:', device)

    for data in  ['davis', 'kiba']:
        # get tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("../hf_models/models--ncfrey--ChemGPT-4.7M/snapshots/7438a282460b3038e17a27e25b85b1376e9a23e2/", local_files_only=True)
        model = AutoModel.from_pretrained("../hf_models/models--ncfrey--ChemGPT-4.7M/snapshots/7438a282460b3038e17a27e25b85b1376e9a23e2/", local_files_only=True)
        
        # get selifes from smile
        selfies = [encoder(s) for s in data]

        # adding a new token '[PAD]' to the tokenizer, and then using it as the padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 

        # stuff that was here before
        cp = torch.load(model_file_name, map_location=device) # loading checkpoint
        model.safe_load_state_dict(cp)
        
        print(f'\n\n{data} model summary:')
        print(model)