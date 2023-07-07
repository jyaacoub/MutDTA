# Models
See `results/model_media/*_stats.csv` for the model statistics for each model.


## DGraphDTA

### Prefix breakdown:
'pretrained_' - performance on PDBbind with just pretrained weights from https://github.com/595693085/DGraphDTA/tree/master/models

* `[n]W_[e]E` - this model was initialized with `n` weights and trained for `e` epochs without MSA data (set to zero matrix) like how it was done in the paper.

* `[n]W_[e]E_msa` - this model was initialized with `n` weights and trained for `e` epochs **with** MSA data as described in the paper (but not done see: https://github.com/595693085/DGraphDTA/issues/15).

* `[n]W_[e]E_msa_shan` - Same as `[n]W_[e]E_msa` but with the shannon entropy instead of PPV for the nodes in the amino acid sequence.
