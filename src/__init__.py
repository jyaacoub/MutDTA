from src.utils import config
from src.utils import config as cfg


TUNED_MODEL_CONFIGS = {
    #DGM_davis0D_nomsaF_binaryE_128B_0.00012LR_0.24D_2000E.model
    'davis_DG':{
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
      	'lr': 0.00012,
    	'batch_size': 128,
    
    	'architecture_kwargs': {
    		'dropout': 0.24,
            'output_dim': 128,
    	}
    }, 
    'davis_aflow':{
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
    #GVPLM_davis0D_nomsaF_aflowE_128B_0.0001360163557088453LR_0.027175922988649594D_2000E_gvpLF_binaryLE
    'davis_gvpl_aflow': {
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
      	'lr': 0.00014968791626986144,
    	'batch_size': 128,
    
    	'architecture_kwargs': {
    		'dropout': 0.00039427600918916277,
    		'output_dim': 256,
    		'num_GVPLayers': 3
    	}
    },
    'davis_esm':{
        "model": cfg.MODEL_OPT.EDI,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
 
        'lr': 0.0001, 
        'batch_size': 48, # global batch size (local was 12)
        
        'architecture_kwargs': {
            'dropout': 0.4, 
            'dropout_prot': 0.0, 
            'output_dim': 128, 
            'pro_extra_fc_lyr': False, 
            'pro_emb_dim': 512 # just for reference since this is the default for EDI
        }        
    },
    'davis_gvpl_esm_aflow': {
        "model": cfg.MODEL_OPT.GVPL_ESM,
                
        "dataset": cfg.DATA_OPT.davis,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
 
        'lr': 0.00010636872718329864, 
        'batch_size': 48, # global batch size (local was 12)
        
        'architecture_kwargs': {
            'dropout': 0.23282479481785903, 
            'output_dim': 512, 
            'num_GVPLayers': 3, 
            'pro_dropout_gnn': 0.15822227777305042, 
            'pro_extra_fc_lyr': False, 
            'pro_emb_dim': 128
        }
    },
    #####################################################
    ############## kiba #################################
    #####################################################
    'kiba_DG': { #DGM_kiba0D_nomsaF_binaryE_128B_0.0001LR_0.4D_2000E
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.kiba,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
      	'lr': 0.0001,
    	'batch_size': 128,
    
    	'architecture_kwargs': {
    		'dropout': 0.4,
            'output_dim': 128,
    	}
    },
    'kiba_aflow':{
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.kiba,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                 
        'lr': 0.0001139464546302261, 
        'batch_size': 64, 
        
        'architecture_kwargs': {
            'dropout': 0.4321620419748407, 
            'output_dim': 512
        }
    },
    'kiba_esm':{ #EDIM_kiba0D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E
        "model": cfg.MODEL_OPT.EDI,
                
        "dataset": cfg.DATA_OPT.kiba,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
 
        'lr': 0.0001, 
        'batch_size': 48, # global batch size (local was 12)
        
        'architecture_kwargs': {
            'dropout': 0.4, 
            'dropout_prot': 0.0, 
            'output_dim': 128, 
            'pro_extra_fc_lyr': False, 
            'pro_emb_dim': 512 # just for reference since this is the default for EDI
        }
    },
    'kiba_gvpl_aflow': {
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.kiba,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
            
        'lr': 0.00010990897170411903, 
        'batch_size': 16, 
        
        'architecture_kwargs': {
            'dropout': 0.03599877069828837, 
            'output_dim': 128, 
            'num_GVPLayers': 2
        }
    },
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
    #DGM_PDBbind0D_nomsaF_binaryE_64B_0.0001LR_0.4D_2000E
    'PDBbind_DG': {
        "model": cfg.MODEL_OPT.DG,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
                
      	'lr': 0.0001,
    	'batch_size': 64,
    
    	'architecture_kwargs': {
    		'dropout': 0.4,
            'output_dim': 128,
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
    #EDIM_PDBbind1D_nomsaF_binaryE_48B_0.0001LR_0.4D_2000E
    'PDBbind_esm':{
        "model": cfg.MODEL_OPT.EDI,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.original,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
 
        'lr': 0.0001, 
        'batch_size': 48, # global batch size (local was 12)
        
        'architecture_kwargs': {
            'dropout': 0.4, 
            'dropout_prot': 0.0, 
            'output_dim': 128, 
            'pro_extra_fc_lyr': False, 
            'pro_emb_dim': 512 # just for reference since this is the default for EDI
        }
    },
    #GVPLM_PDBbind0D_nomsaF_aflowE_128B_0.00022659LR_0.02414D_2000E_gvpLF_binaryLE
    'PDBbind_gvpl_aflow':{
        "model": cfg.MODEL_OPT.GVPL,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.aflow,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
            
        'lr': 0.00020048122460779208, 
        'batch_size': 128, 
        
        'architecture_kwargs': {
            'dropout': 0.042268679447260635, 
            'output_dim': 512,
            'num_GVPLayers': 3,
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
    'PDBbind_gvpl_esm':{
        "model": cfg.MODEL_OPT.GVPL_ESM,
                
        "dataset": cfg.DATA_OPT.PDBbind,
        "feature_opt": cfg.PRO_FEAT_OPT.nomsa,
        "edge_opt": cfg.PRO_EDGE_OPT.binary,
        "lig_feat_opt": cfg.LIG_FEAT_OPT.gvp,
        "lig_edge_opt": cfg.LIG_EDGE_OPT.binary,
 
        'lr': 0.0001, 
        'batch_size': 40, # global batch size (local is 10)
        
        'architecture_kwargs': {
            'dropout': 0.2328, 
            'output_dim': 128, 
            'num_GVPLayers': 3, 
            'pro_dropout_gnn': 0.1582, 
            'pro_extra_fc_lyr': False, 
            'pro_emb_dim': 512,
        }
    }
}
