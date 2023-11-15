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
score_history_f1 = [folds[0][2]]
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
        fold[2] = fold[1] - abs(fold[1]/f_len - c) # without this term it performs better
        
    # Finding optimal fold to add protein to (minimize score)
    best_fold = min(folds, key=lambda x: x[2])
    
    # Add protein to fold
    best_fold[0].append(p)
    # update weight
    best_fold[1] += c
    
    # update score history
    score_history_f1.append(folds[0][2])



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
# %% Plotting distributions of each fold over each other
import matplotlib.pyplot as plt
import seaborn as sns

# x-axis will be the protein counts
# y-axis will be the number of proteins with that count

# get protein counts for each fold
counts = [Counter(df[df['prot_id'].isin(f[0])]['prot_id']) for f in folds]

# %% plot
bin_range = range(0, 100, 2)
kde = True
if data == 'kiba':
    bin_range = range(0, 1400, 100)
elif data == 'davis0':
    bin_range = [67,69] # 68 is the only count for davis
    kde = False


ax = sns.histplot(counts, stat='count', bins=bin_range, kde=kde, alpha=0.3,
                  linewidth=0)
# limit x-axis to 100
if data == 'pdb':
    ax.set_xlim([0, 20])
ax.set_title(f'Protein Count Distribution for {data.upper()}')
ax.set_xlabel('Count of protein')
ax.set_ylabel('Frequency')
plt.show()

# %%
