import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from src.utils.residue import ResInfo, Chain
from src.utils.mutate_model import run_modeller
import copy

def plot_sequence(muta, pro_seq, pep_opts=ResInfo.amino_acids[:-1], delta=False):
    if delta:
        muta = copy.deepcopy(muta)
        original_pkd = None
        for i, AA in enumerate(pep_opts):
            if AA == pro_seq[0]:
                original_pkd = muta[i,0]
            
        muta -= original_pkd
                
    
    # Create the plot
    plt.figure(figsize=(len(pro_seq), 20))  # Adjust the figure size as needed
    plt.imshow(muta, aspect='auto', cmap='coolwarm', interpolation='none')

    # Set the x-ticks to correspond to positions in the protein sequence
    plt.xticks(ticks=np.arange(len(pro_seq)), labels=pro_seq, fontsize=16)
    plt.yticks(ticks=np.arange(len(pep_opts)), labels=pep_opts, fontsize=16)
    plt.xlabel('Original Protein Sequence', fontsize=75)
    plt.ylabel('Mutated to Amino Acid code', fontsize=75)

    # Add text labels to each square
    for i in range(len(pep_opts)):
        for j in range(len(pro_seq)):
            text = plt.text(j, i, f'{pep_opts[i]}', ha='center', va='center', color='black', fontsize=12)
            # Add a white outline to the text
            text.set_path_effects([
                PathEffects.Stroke(linewidth=1, foreground='white'),
                PathEffects.Normal()
            ])


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
    res_range = (0,215)
    muta = np.load(f'muta-{res_range[0]}_{res_range[1]}.npy')
    #plot_sequence(muta[:,res_range[0]:res_range[1]], 
    #              pro_seq=Chain(pdb_file).sequence[res_range[0]:res_range[1]])




