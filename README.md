# MutDTA
Improving the precision oncology pipeline by providing binding affinity purtubations predictions on a pirori identified cancer driver genes.

# Current Progress
- [ ] Data preprocessing
  - [x] PDBbind - simple enough to use. Protein seq were downloaded from UniProt.
  - [x] PLATINUM - Working on getting mutated sequence data (done with mutalyzer)
  - [ ] KIBA and Davis - Kinase proteins are not super relevant but could be useful for pretraining since datasets are limited.
  - [ ] GENIE - This will require some physical docking methods since we have no binding affinity data for this.
- [x] Docking baseline
  - [x] Set up Docking on cluster
  - [x] Build scripts to automate ligand and protein prep (including grid for binding site).
  - [x] Run docking on PDBbind dataset
- [ ] Model baseline
  - [ ] DGraphDTA
       - [x] Evaluate pretrained model on PDBbind dataset
       - [x] Train model on *refined-set* PDBbind dataset and evaluate
       - [ ] Train model on *general* PDBbind dataset and evaluate

# AutoDock Vina Procedure
See [README/VINA_PROCEDURE.md](./docs/VINA_PROCEDURE.MD) for detailed steps

## Contribution
See: https://gist.github.com/Zekfad/f51cb06ac76e2457f11c80ed705c95a3 for conventional commits cheat sheet.