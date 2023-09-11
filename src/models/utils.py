from torch import nn
from src.utils.loader import Loader

class BaseModel(nn.Module):
    """
    Base model for printing summary
    """
    def __init__(self, pro_feat, edge_weight_opt, *args, **kwargs) -> None:
        edge_weight_opt = edge_weight_opt or 'binary' # None -> binary
        assert edge_weight_opt in Loader.edge_opt
        self.edge_weight = not (edge_weight_opt == 'binary')
        self.esm_only = pro_feat == 'esm_only'
        
        super().__init__(*args, **kwargs)
    
    def __str__(self) -> str:
        main_str = super().__str__()
        # model size
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2


        main_str += f'\nmodel size: {size_all_mb:.3f}MB'
        return main_str
    
