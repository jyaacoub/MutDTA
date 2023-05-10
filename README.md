# MutDTA
Improving the precision oncology pipeline by providing binding affinity purtubations predictions on a pirori identified cancer driver genes.

# Current Progress
- [ ] Data preprocessing
  - [x] PDBbind - simple enough to use. Protein seq were downloaded from UniProt.
  - [ ] PLATINUM - Working on getting mutated sequence data
  - [ ] KIBA and Davis - Kinase proteins are not super relevant but could be useful for pretraining since datasets are limited.
  - [ ] GENIE - This will require some physical docking methods since we have no binding affinity data for this.
- [ ] Model training
- [ ] Model evaluation