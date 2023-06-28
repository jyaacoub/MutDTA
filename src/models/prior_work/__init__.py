from src.models.prior_work.models import DGraphDTA


def display_models():
    import torch
    from torch_geometric.nn import summary
    from torch_geometric import data as geo_data

    cuda_name = 'cuda:0'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('cuda_name:', cuda_name)
    print('device:', device)

    for data in  ['davis', 'kiba']:
        model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{data}_t2.model'
        model = DGraphDTA()
        model.to(device)
        
        cp = torch.load(model_file_name, map_location=device) # loading checkpoint
        model.load_state_dict(cp)
        
        print(f'\n\n{data} model summary:')
        print(model)
