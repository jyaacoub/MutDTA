from torch import nn
from src.utils import config as cfg

class BaseModel(nn.Module):
    """
    Base model for printing summary
    """
    def __init__(self, pro_feat, edge_weight_opt, *args, **kwargs) -> None:
        edge_weight_opt = edge_weight_opt or 'binary' # None -> binary
        assert edge_weight_opt in cfg.EDGE_OPT
        self.edge_weight = not (edge_weight_opt == 'binary')
        self.esm_only = (pro_feat == 'esm_only') if pro_feat is not None else False
        
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
    
    def safe_load_state_dict(self, mdl_dict) -> any:
        """Error catching version of `load_state_dict` to resolve Exceptions due to `module.` 
        prefix added by DDP

        Args:
            mdl_dict (_type_): torch.load output for loaded model
        """
        self.load_state_dict(mdl_dict)
        try:
            self.load_state_dict(mdl_dict)
        except RuntimeError as e:
            # if model was distributed then it will have extra "module." prefix
            # due to https://discuss.pytorch.org/t/check-if-model-is-wrapped-in-nn-dataparallel/67957
            # print("Error(s) in loading state_dict for EsmDTA")
            mdl_dict = {(k[7:] if 'module.' == k[:7] else k):v for k,v in mdl_dict.items()}
            self.load_state_dict(mdl_dict)
        
    
    
