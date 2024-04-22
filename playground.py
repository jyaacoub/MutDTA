#%%
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to load sequences and their lengths from csv files
def load_sequences(directory):
    lengths = []
    labels_positions = {}  # Dictionary to hold the last length of each file for labeling
    files = sorted([f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('input_')])
    for file in files:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path)
        # Extract lengths
        current_lengths = data['seqres'].apply(len)
        lengths.extend(current_lengths)
        # Store the position for the label using the last length in the current file
        labels_positions[int(file.split('_')[1].split('.')[0])] = current_lengths.iloc[0]
    return lengths, labels_positions

p = lambda d: f"/cluster/home/t122995uhn/projects/data/{d}/alphaflow_io"

DATASETS = {d: p(d) for d in ['davis', 'kiba', 'pdbbind']}
DATASETS['platinum'] = "/cluster/home/t122995uhn/projects/data/PlatinumDataset/raw/alphaflow_io"

fig, axs = plt.subplots(len(DATASETS), 1, figsize=(10, 5*len(DATASETS) + len(DATASETS)))

n_bins = 50  # Adjust the number of bins according to your preference

for i, (dataset, d_dir) in enumerate(DATASETS.items()):
    # Load sequences and positions for labels
    lengths, labels_positions = load_sequences(d_dir)
    
    # Plot histogram
    ax = axs[i]
    n, bins, patches = ax.hist(lengths, bins=n_bins, color='blue', alpha=0.7)
    ax.set_title(dataset)
    
    # Add counts to each bin
    for count, x, patch in zip(n, bins, patches):
        ax.text(x + 0.5, count, str(int(count)), ha='center', va='bottom')
    
    # Adding red number labels
    for label, pos in labels_positions.items():
        ax.text(pos, label, str(label), color='red', ha='center')
    
    # Optional: Additional formatting for readability
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, max(lengths) + 10])  # Adjust xlim to make sure labels fit

plt.tight_layout()
plt.show()
# %%
