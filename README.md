# Requirements
* AutoDock Tools script and pythonsh from MGL
* `.bashrc` with relevant aliases for molecule prep scripts like `prepare_ligand4.py`:

```bash
export PATH=~/mgltools_x86_64Linux2_1.5.7/bin:$PATH #

alias pmv='~/mgltools_x86_64Linux2_1.5.7/bin/pmv'           # not neccessary
alias adt='~/mgltools_x86_64Linux2_1.5.7/bin/adt'           # not neccessary
alias vision='~/mgltools_x86_64Linux2_1.5.7/bin/vision'     # not neccessary
alias pythonsh='~/mgltools_x86_64Linux2_1.5.7/bin/pythonsh' # important

alias prep_prot='pythonsh ~/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'   # important
alias prep_lig='pythonsh ~/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'      # important
```
