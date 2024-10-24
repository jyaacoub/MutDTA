"""
This script contains some functions for transforming different architecture state dicts into the updated state dicts
- This is needed since some achitecture changes occured since training causing GVPL models to not be able to load previous state dicts:
- We removed unused parameters thats why we see a bunch of unexpected keys in the state dict below (dummy params and unused fc1)

```
Traceback (most recent call last):
  File "/lustre06/project/6069023/jyaacoub/MutDTA/run_mutagenesis.py", line 102, in <module>
    m, _ = Loader.load_tuned_model(MODEL_OPT, fold=FOLD, device=DEVICE)
  File "/lustre06/project/6069023/jyaacoub/MutDTA/src/utils/loader.py", line 28, in wrapper
    return func(*args, **kwargs)
  File "/lustre06/project/6069023/jyaacoub/MutDTA/src/utils/loader.py", line 112, in load_tuned_model
    model.load_state_dict(torch.load(model_p, map_location=device))
  File "/lustre06/project/6069023/jyaacoub/MutDTA/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2041, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for GVPLigand_DGPro:
	Missing key(s) in state_dict: "pro_fc.0.weight", "pro_fc.0.bias", "pro_fc.3.weight", "pro_fc.3.bias". 
	Unexpected key(s) in state_dict: 
  "pro_fc_g1.weight", "pro_fc_g1.bias", "pro_fc_g2.weight", "pro_fc_g2.bias", # these are the only ones we care about all the rest are unused
  
  "mol_conv1.bias", "mol_conv1.lin.weight", "mol_conv2.bias", "mol_conv2.lin.weight", "mol_conv3.bias", "mol_conv3.lin.weight", "mol_fc_g1.weight", 
  "mol_fc_g1.bias", "mol_fc_g2.weight", "mol_fc_g2.bias", 
  "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
  "out.weight", "out.bias",
  "gvp_ligand.W_v.1.dummy_param", "gvp_ligand.W_e.1.dummy_param", "gvp_ligand.layers.0.conv.message_func.0.dummy_param", 
  "gvp_ligand.layers.0.conv.message_func.1.dummy_param", "gvp_ligand.layers.0.conv.message_func.2.dummy_param", "gvp_ligand.layers.0.dropout.0.vdropout.dummy_param", 
  "gvp_ligand.layers.0.dropout.1.vdropout.dummy_param", "gvp_ligand.layers.0.ff_func.0.dummy_param", "gvp_ligand.layers.0.ff_func.1.dummy_param",  "gvp_ligand.layers.1.conv.message_func.0.dummy_param", 
    "gvp_ligand.layers.1.conv.message_func.1.dummy_param", 
    "gvp_ligand.layers.1.conv.message_func.2.dummy_param", 
    "gvp_ligand.layers.1.dropout.0.vdropout.dummy_param", 
    "gvp_ligand.layers.1.dropout.1.vdropout.dummy_param", 
    "gvp_ligand.layers.1.ff_func.0.dummy_param", 
    "gvp_ligand.layers.1.ff_func.1.dummy_param", 
    "gvp_ligand.layers.2.conv.message_func.0.dummy_param", 
    "gvp_ligand.layers.2.conv.message_func.1.dummy_param", 
    "gvp_ligand.layers.2.conv.message_func.2.dummy_param", 
    "gvp_ligand.layers.2.dropout.0.vdropout.dummy_param", 
    "gvp_ligand.layers.2.dropout.1.vdropout.dummy_param", 
    "gvp_ligand.layers.2.ff_func.0.dummy_param",
    "gvp_ligand.layers.2.ff_func.1.dummy_param", "gvp_ligand.W_out.1.dummy_param". 
```
"""

def GVPLigand_DGPro_transform(sd:dict):
    # to remove:
    remove = [
        "mol_conv1.bias", "mol_conv1.lin.weight", 
        "mol_conv2.bias", "mol_conv2.lin.weight", 
        "mol_conv3.bias", "mol_conv3.lin.weight", 
        "mol_fc_g1.weight", "mol_fc_g1.bias", 
        "mol_fc_g2.weight", "mol_fc_g2.bias", 
        "fc1.weight", "fc1.bias", 
        "fc2.weight", "fc2.bias",
        "out.weight", "out.bias",
        "gvp_ligand.W_v.1.dummy_param", "gvp_ligand.W_e.1.dummy_param", 
        "gvp_ligand.layers.0.conv.message_func.0.dummy_param", "gvp_ligand.layers.0.conv.message_func.1.dummy_param", "gvp_ligand.layers.0.conv.message_func.2.dummy_param", 
        "gvp_ligand.layers.0.dropout.0.vdropout.dummy_param", "gvp_ligand.layers.0.dropout.1.vdropout.dummy_param", 
        "gvp_ligand.layers.0.ff_func.0.dummy_param", "gvp_ligand.layers.0.ff_func.1.dummy_param",  
        "gvp_ligand.layers.1.conv.message_func.0.dummy_param", "gvp_ligand.layers.1.conv.message_func.1.dummy_param", "gvp_ligand.layers.1.conv.message_func.2.dummy_param", 
        "gvp_ligand.layers.1.dropout.0.vdropout.dummy_param", "gvp_ligand.layers.1.dropout.1.vdropout.dummy_param", 
        "gvp_ligand.layers.1.ff_func.0.dummy_param", "gvp_ligand.layers.1.ff_func.1.dummy_param", 
        "gvp_ligand.layers.2.conv.message_func.0.dummy_param", "gvp_ligand.layers.2.conv.message_func.1.dummy_param", "gvp_ligand.layers.2.conv.message_func.2.dummy_param", 
        "gvp_ligand.layers.2.dropout.0.vdropout.dummy_param", "gvp_ligand.layers.2.dropout.1.vdropout.dummy_param", 
        "gvp_ligand.layers.2.ff_func.0.dummy_param","gvp_ligand.layers.2.ff_func.1.dummy_param", 
        "gvp_ligand.W_out.1.dummy_param"
    ]
    for k in remove:
        del sd[k]
        
    # renaming:
    sd["pro_fc.0.weight"]   = sd.pop("pro_fc_g1.weight")
    sd["pro_fc.0.bias"]     = sd.pop("pro_fc_g1.bias")
    sd["pro_fc.3.weight"]   = sd.pop("pro_fc_g2.weight") # linear layer is the third in the sequential call
    sd["pro_fc.3.bias"]     = sd.pop("pro_fc_g2.bias")
    
    return sd
    
    
    
    
    