# H4H Directory structure and where to find things:

## H4H:

Everything important (e.g.: model weights) should be accessible through the shared project directory.

I primarily used my home directory to store the code since I could sync it up with GitHub whereas on the shared project directory we have no internet access to perform “git pull/push” operations. 

Thus for files/folders that were too large to store in HOME, I used symbolic links to folders located in the shared project directory. However, I forget exactly how I laid everything out and I no longer have access to the VPN to connect and check. 

Nonetheless, all the important stuff we would need, like model checkpoints, should be stored in the shared project directory.

## GitHub \- [https://github.com/jyaacoub/MutDTA/tree/main](https://github.com/jyaacoub/MutDTA/tree/main)

[Training splits](https://github.com/jyaacoub/MutDTA/tree/main/splits) can be found on the GitHub page as well as all my most recent code.

# Training on new datasets or new models
1. For training on existing data you would use the [train_folds.sh](https://github.com/jyaacoub/MutDTA/blob/main/SBATCH/train_folds.sh) script, depending on your comfort with editing the existing python scripts it might be a bit difficult to set up. But you just need to define a new "model_opt" in [src/utils/loader.py](https://github.com/jyaacoub/MutDTA/blob/99921771e31349d8a9564be6aec9fdab35ce0ae6/src/utils/loader.py#L146), and add that model key to the list of options in [src/utils/config.py](https://github.com/jyaacoub/MutDTA/blob/99921771e31349d8a9564be6aec9fdab35ce0ae6/src/utils/config.py#L25).
2. If you make any changes to the input model features this would make things a lot harder since this is essentially building a new dataset with those features and would need to add instructions on how to set that up for [protein features](https://github.com/jyaacoub/MutDTA/blob/99921771e31349d8a9564be6aec9fdab35ce0ae6/src/data_prep/feature_extraction/protein.py#L59), [protein edges](https://github.com/jyaacoub/MutDTA/blob/99921771e31349d8a9564be6aec9fdab35ce0ae6/src/data_prep/feature_extraction/protein_edges.py#L10), and [ligand features](https://github.com/jyaacoub/MutDTA/blob/main/src/data_prep/feature_extraction/ligand.py).
3. For entirely new datasets this is more challenging since you basically need to build a new Dataset subclass (inherited from the BaseDataset class) - see [PlatinumDataset](https://github.com/jyaacoub/MutDTA/blob/99921771e31349d8a9564be6aec9fdab35ce0ae6/src/data_prep/datasets.py#L1019) for a good example on this (it is the cleanest of the 3 dataset classes I have).

# GitHub issues

All the issues we encountered with this project are tracked via GitHub. I list some of the more relevant issues below:

## Summary of model checkpoints/issues (found in [MutDTA/results/](https://github.com/jyaacoub/MutDTA/tree/main/results)):

Basically the only ones that matter are `results/model_checkpoints` and `v103`. The rest are just some tests I did to resolve/debug issues.

* [`results/model_checkpoints`](https://github.com/jyaacoub/MutDTA/tree/main/results)  \- These are the models trained on *random splits*  
* [`v103`](https://github.com/jyaacoub/MutDTA/issues/103) \- **pocket-only** representation checkpoints   
* [`v113`](https://github.com/jyaacoub/MutDTA/issues/113) \- new training split where we excluded highly targeted (OncoKB) proteins from training.  
  * This leads to consistently worse performance across the board.  
* [`v115`](https://github.com/jyaacoub/MutDTA/issues/115)  \- since "aflow" (alphaflow edge weights) models had a smaller dataset (due to memory issues when running Alphaflow on AA sequences 1200+) we *artificially reduced the sizes of the training sets* for the other models so that we could have a fair comparison  
  * This didn't change much.  
* [`v128`](https://github.com/jyaacoub/MutDTA/issues/128)   \- Test to see if new splits were the issue with weirdly low performance with oncoKB split (they were)

## OncoKB distribution drift issue with splits \- [Issue \#131](https://github.com/jyaacoub/MutDTA/issues/131)

When we originally started looking into OncoKB I selected highly targeted proteins from OncoKB to be excluded from training sets.

- This caused a big distribution drift issue and resulted in much worse performance, particularly with PDBbind.

Stats on the distribution differences between the manually curated oncokb dataset split vs a random split can be found on the [issue page.](https://github.com/jyaacoub/MutDTA/issues/131#issuecomment-2276366754)

- click the *details* button to see figures.

## Missing Amino Acids in PDBs for PDBbind \- [Issue\#102](https://github.com/jyaacoub/MutDTA/issues/102)

This means for the pocket versions of our models we can’t readily use existing scripts to get the pocket sequence graph based on the PDBs provided.

- It is possible to fix this, but it needs a LOT of effort since we would also need to retrain the PDBbind models that used graphs with the missing residues.

## Pocket representation version of our models \- [Issue\#103](https://github.com/jyaacoub/MutDTA/issues/103)

This tracks how the pocket representation of Davis and Kiba models was built. The [pull request 135](https://github.com/jyaacoub/MutDTA/pull/135) resolves this with the results in the [CSV files](https://github.com/jyaacoub/MutDTA/pull/135/files#diff-470793793283a1e1b2c3c5055749ddb946413c66b5581a70bb502db544660642).

## 
