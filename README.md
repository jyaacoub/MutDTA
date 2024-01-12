# MutDTA [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/) ![Python Tests](https://github.com/jyaacoub/MutDTA/actions/workflows/python-app.yml/badge.svg?branch=main) 
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
Try to follow [conventional commits](https://gist.github.com/Zekfad/f51cb06ac76e2457f11c80ed705c95a3).

### Quick examples
* `feat: new feature`
* `fix(scope): bug in scope`
* `feat!: breaking change` / `feat(scope)!: rework API`
* `chore(deps): update dependencies`

### Commit types
* `build`: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
* `ci`: Changes to CI configuration files and scripts (example scopes: Travis, Circle, BrowserStack, SauceLabs)
* **`chore`: Changes which doesn't change source code or tests e.g. changes to the build process, auxiliary tools, libraries**
* `docs`: Documentation only changes
* **`feat`: A new feature**
* **`fix`: A bug fix**
* `perf`: A code change that improves performance
* `refactor`:  A code change that neither fixes a bug nor adds a feature
* `revert`: Revert something
* `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
* `test`: Adding missing tests or correcting existing tests

### Reminders
* Put newline before extended commit body
* More details at **[conventionalcommits.org](https://www.conventionalcommits.org/)**
