from src.utils import config
from src.utils import config as cfg


TUNED_MODEL_CONFIGS = {
    #GVPLM_davis0D_nomsaF_aflowE_128B_0.0001360163557088453LR_0.027175922988649594D_2000E_gvpLF_binaryLE
    'davis_gvpl_aflow': {
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
        "lr": 0.0001360163557088453,
        "batch_size": 128, # local batch size
        
        "architecture_kwargs":{
            "dropout": 0.027175922988649594,
            "output_dim":  128,
        }
    },
    #GVPLM_davis3D_nomsaF_binaryE_128B_0.00020535607176845963LR_0.08845592454543601D_2000E_gvpLF_binaryLE
    'davis_gvpl': {
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
        'lr': 0.00020535607176845963, 
        'batch_size': 128, 
        'architecture_kwargs': {
            'dropout': 0.08845592454543601, 
            'output_dim': 512
        }
    },
    
    'davis_aflow':{ # not trained yet...
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
            
        'lr': 0.0008279387625584954, 
        'batch_size': 128, 
        
        'architecture_kwargs': {
            'dropout': 0.3480347297724069, 
            'output_dim': 256
        }
    },
    #####################################################
    ############## kiba #################################
    #####################################################
    'kiba_gvpl': {
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.kiba,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
            
        'lr': 0.00003372637625954074, 
        'batch_size': 32, 
        
        'architecture_kwargs': {
            'dropout': 0.09399264336737133,
            'output_dim': 512, 
            'num_GVPLayers': 4
        }
    },
    
    
    #####################################################
    ########### PDBbind #################################
    #####################################################
    #GVPLM_PDBbind0D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE
    'PDBbind_gvpl_aflow':{
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
            
        'lr': 0.00022659, 
        'batch_size': 128, 
        
        'architecture_kwargs': {
            'dropout': 0.02414, 
            'output_dim': 256
        }
    },
    'PDBbind_gvpl':{
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
            
        'lr': 0.00020066831190641135, 
        'batch_size': 128, 
        
        'architecture_kwargs': {
            'dropout': 0.4661593536060576, 
            'output_dim': 512
        }
    },
    'PDBbind_aflow':{
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
            
        'lr': 0.0009185598967356679, 
        'batch_size': 128, 
        
        'architecture_kwargs': {
            'dropout': 0.22880989869337157, 
            'output_dim': 256
        }
    },
}
