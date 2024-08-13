import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from src.utils.residue import ResInfo, Chain
from src.utils.mutate_model import run_modeller


def plot_sequence(muta, pep_opts, pro_seq):
    # Create the plot
    plt.figure(figsize=(len(pro_seq), 20))  # Adjust the figure size as needed
    plt.imshow(muta, aspect='auto', cmap='coolwarm', interpolation='none')

    # Set the x-ticks to correspond to positions in the protein sequence
    plt.xticks(ticks=np.arange(len(pro_seq)), labels=[ResInfo.code_to_pep[p] for p in pro_seq], rotation=45, fontsize=16)
    plt.yticks(ticks=np.arange(len(pep_opts)), labels=pep_opts, fontsize=16)
    plt.xlabel('Protein Sequence Position', fontsize=75)
    plt.ylabel('Peptide Options', fontsize=75)

    # Add text labels to each square
    for i in range(len(pep_opts)):
        for j in range(len(pro_seq)):
            text = plt.text(j, i, f'{ResInfo.pep_to_code[pep_opts[i]]}', ha='center', va='center', color='black', fontsize=8)
            # Add a white outline to the text
            text.set_path_effects([
                PathEffects.Stroke(linewidth=1, foreground='white'),
                PathEffects.Normal()
            ])
            break


    # Adjust gridlines to be off-center, forming cell boundaries
    plt.gca().set_xticks(np.arange(-0.5, len(pro_seq), 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, len(pep_opts), 1), minor=True)
    plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Remove the major gridlines (optional, for clarity)
    plt.grid(which='major', color='none')

    # Add a colorbar to show the scale of mutation values
    plt.colorbar(label='Mutation Values')
    plt.title('Mutation Array Visualization', fontsize=100)
    plt.show()



if __name__ == "__main__":
    pro_id = "P67870"
    pdb_file = f'/cluster/home/t122995uhn/projects/tmp/kiba/{pro_id}.pdb'


    plot_sequence(np.load('muta-100_200.npy'), pep_opts=list(ResInfo.pep_to_code.keys()),
                pro_seq=Chain(pdb_file).sequence)
    
    
    # %%
    import os
    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    import numpy as np
    import torch
    import torch_geometric as torchg
    from tqdm import tqdm

    from src import cfg
    from src.utils.loader import Loader
    from src.utils.residue import ResInfo, Chain
    from src.data_prep.feature_extraction.ligand import smile_to_graph
    from src.data_prep.feature_extraction.protein import target_to_graph
    from src.utils.residue import ResInfo, Chain

    #%%
    DATA = cfg.DATA_OPT.kiba
    lig_feature = cfg.LIG_FEAT_OPT.original
    lig_edge = cfg.LIG_EDGE_OPT.binary
    pro_feature = cfg.PRO_FEAT_OPT.nomsa
    pro_edge = cfg.PRO_EDGE_OPT.binary

    lig_seq = "COC1=C(C=CC(=C1)C2=CC3=C(C=C2)C(=CC4=CC=CN4)C(=O)N3)O" #CHEMBL202930

    # %% build ligand graph
    mol_feat, mol_edge = smile_to_graph(lig_seq, lig_feature=lig_feature, lig_edge=lig_edge)
    lig = torchg.data.Data(x=torch.Tensor(mol_feat), edge_index=torch.LongTensor(mol_edge),lig_seq=lig_seq)

    # %% Get initial pkd value:
    pro_id = "P67870"
    pdb_file = f'/cluster/home/t122995uhn/projects/tmp/kiba/{pro_id}.pdb'

    def get_protein_features(pro_id, pdb_file, DATA=DATA):
        pdb = Chain(pdb_file)
        pro_seq = pdb.sequence
        pro_cmap = pdb.get_contact_map()

        updated_seq, extra_feat, edge_idx = target_to_graph(target_sequence=pro_seq, 
                                                            contact_map=pro_cmap,
                                                            threshold=8.0 if DATA is cfg.DATA_OPT.PDBbind else -0.5, 
                                                            pro_feat=pro_feature)
        pro_feat = torch.Tensor(extra_feat)

        pro = torchg.data.Data(x=torch.Tensor(pro_feat),
                            edge_index=torch.LongTensor(edge_idx),
                            pro_seq=updated_seq, # Protein sequence for downstream esm model
                            prot_id=pro_id,
                            edge_weight=None)
        return pro, pro_seq

    pro, pro_seq = get_protein_features(pro_id, pdb_file)

    # %% Loading the model
    m = Loader.load_tuned_model('davis_DG', fold=1)
    m.eval()
    original_pkd = m(pro, lig)
    print(original_pkd)

    # %% mutate and regenerate graphs
    muta = np.zeros(shape=(len(ResInfo.pep_to_code.keys()), len(pro_seq)))

    # zero indexed res range to mutate:
    res_range = (100, 200)
    res_range = (max(res_range[0], 0),
                min(res_range[1], len(pro_seq)))

    # %%
    from src.utils.mutate_model import run_modeller

    amino_acids = ResInfo.amino_acids[:-1] # not including "X" - unknown

    with tqdm(range(*res_range), ncols=80, total=(res_range[1]-res_range[0])) as t:
        for j in t:
            for i, AA in enumerate(amino_acids):
                if i%2 == 0:
                    t.set_postfix(res=j, AA=i+1)
                
                if pro_seq[i] == AA:
                    muta[i,j] = original_pkd
                    continue
                    
                pro_id = "P67870"
                pdb_file = f'/cluster/home/t122995uhn/projects/tmp/kiba/{pro_id}.pdb'
                out_pdb_fp = run_modeller(pdb_file, 1, ResInfo.code_to_pep[AA], "A")
                
                pro, _ = get_protein_features(pro_id, out_pdb_fp)
                muta[i,j] = m(pro, lig)
                
                # delete after use
                os.remove(out_pdb_fp)
                
    #%%
    np.save(f"muta-{res_range[0]}_{res_range[1]}.npy", muta)



