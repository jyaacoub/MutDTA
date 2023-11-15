# %%
from collections import Counter, OrderedDict
import pandas as pd

data = 'davis0'
data = 'kiba'
data = 'pdb'
df = pd.read_csv(f'../data/misc/{data}_XY.csv', index_col=0)

#%%
prot_counts = Counter(df['prot_id'])

#%%
########## split remaining proteins into k_folds ##########
# Steps for this basically follow Greedy Number Partitioning
# 1. Initialize variables:
#       - folds = list of lists of protein ids
#       - prots = list of proteins  
# 2. Sort protein counts by number of samples
k = 5

folds = [[[], 0, -1] for i in range(k)] # tuple of (list of proteins, total weight, current-score)
#   - score = fold.weight - abs(fold.weight/len(fold) - item.weight)
# the most optimal data structure for folds is a list since their score 
# must be updated every time a protein is added to a fold
counts_sorted = sorted(list(prot_counts.items()), key=lambda x: x[1], reverse=True)
for p, c in counts_sorted:
    # Update scores for each fold
    for fold in folds:
        f_len = len(fold[0])
        if f_len == 0:
            continue
        
        # calculate score for adding protein to fold
        fold[2] = fold[1] #- abs(fold[1]/f_len - c)
        
    # Finding optimal fold to add protein to (minimize score)
    best_fold = min(folds, key=lambda x: x[2])
    
    # Add protein to fold
    best_fold[0].append(p)
    # update weight
    best_fold[1] += c



# %% convert folds to set after done selecting for faster lookup
folds_sets = [set(f[0]) for f in folds]

# validating that they dont intersect with each other
for i in range(len(folds_sets)):
    for j in range(i+1, len(folds_sets)):
        assert len(folds_sets[i].intersection(folds_sets[j])) == 0, "Folds intersect"

print("\t\tDataset:", data.upper())
print(f'{"#":>10} | {"num_prots":^10} | {"total_count":^12} | {"final_score":^10}')
print('-'*53)
for i,f in enumerate(folds): print(f'{"Fold "+str(i):>10} | {len(f[0]):^10} | {f[1]:^12} | {f[2]:^10}')
# %%
