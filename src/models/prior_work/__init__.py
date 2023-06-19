from src.models.prior_work.models import DGraphDTA


def display_models():
    import torch
    from torch_geometric.nn import summary
    from torch_geometric import data as geo_data

    for data in  ['davis', 'kiba']:
        model_file_name = f'results/model_checkpoints/prior_work/DGraphDTA_{data}_t2.model'
        cuda_name = 'cuda:0'
        device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
        print('cuda_name:', cuda_name)
        print('device:', device)

        # %%
        model = DGraphDTA()
        model.to(device)
        
        cp = torch.load(model_file_name, map_location=device) # loading checkpoint
        model.load_state_dict(cp)
        # %% Print model summary
        prot = geo_data.Data(x=torch.Tensor(54,54), # node feature matrix
                            edge_index=torch.LongTensor([[0,1]]).transpose(1, 0),
                            y=torch.FloatTensor([1])).to(device)
        lig = geo_data.Data(x=torch.Tensor(78,78), # node feature matrix
                            edge_index=torch.LongTensor([[0,1]]).transpose(1, 0),
                            y=torch.FloatTensor([1])).to(device)
        model_summary = summary(model, lig, prot)

        # model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        # %%
        print(f'Model: {model_file_name}')
        print(model_summary)
        print('model size: {:.3f}MB'.format(size_all_mb))
