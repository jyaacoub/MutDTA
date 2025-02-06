#!/bin/bash
cd /lustre06/project/6069023/jyaacoub/MutDTA/
source .venv/bin/activate

numpy_matrix_fp="/lustre06/project/6069023/jyaacoub/MutDTA/SBATCH/outs/mutagenesis_tests/1a30_ligand/davis_DG/0_5.npy"
pdb_file="/lustre06/project/6069023/jyaacoub/MutDTA/SBATCH/samples/input/mutagenesis/P67870.pdb"
output_image="out.png"
res_start=0
res_end=1000


python -u << EOF
from matplotlib import pyplot as plt
import numpy as np
from src.analysis.mutagenesis_plot import plot_sequence
from src.utils.residue import Chain

muta = np.load('${numpy_matrix_fp}')
pdb_file = '${pdb_file}'

# Plots the full sequence:
# plot_sequence(muta, pro_seq=Chain(pdb_file).sequence)


# Plot a specific residue range (e.g.: a pocket)
res_range = (${res_start}, ${res_end})
plot_sequence(muta[:,res_range[0]:res_range[1]], pro_seq=Chain(pdb_file).sequence[res_range[0]:res_range[1]])

plt.savefig('${output_image}')
EOF