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
# AutoDock Vina Procedure

## 1. Install AutoDock Vina
Can install using wget and unzip in Linux (see https://vina.scripps.edu/downloads/ for other OS):
```bash
wget https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz
```
```bash
tar -xvzf autodock_vina_1_1_2_linux_x86.tgz
```
Directory should match the following
```bash
autodock_vina_1_1_2_linux_x86/LICENSE
autodock_vina_1_1_2_linux_x86/bin/
autodock_vina_1_1_2_linux_x86/bin/vina
autodock_vina_1_1_2_linux_x86/bin/vina_split
```

Run the following to test if it works:
```bash
./autodock_vina_1_1_2_linux_x86/bin/vina --help
```

## 2. Prepare Receptor and Ligand

For preparing the receptor and ligand we will use AutoDock Tools (ADT) from MGL. ADT is a GUI for preparing the receptor and ligand, however it comes with python scripts that we can run from the cmd line. Also it includes a prebuilt Python 2.0 interpreter called `pythonsh` that we can use to run the scripts.

### **2.1 Installing ADT**
Download ADT from http://mgltools.scripps.edu/downloads/ and unzip. 
```bash
wget https://ccsb.scripps.edu/mgltools/download/491/mgltools_x86_64Linux2_1.5.7p1.tar.gz
```
```bash
tar -xvzf mgltools_x86_64Linux2_1.5.7p1.tar.gz
```
Run install script
```bash
mgltools_x86_64Linux2_1.5.7/install.sh
```
>NOTE if you get the following error:
>```bash
>./install.sh: 79: export: (x86)/Common: bad variable name
>```
>This is because mgl automatically trys to add the `bin` directory to the root,, just uncomment that line (line 79) and you should be good.

Now we have `pythonsh` and ADT installed, they should be located in:
```bash
mgltools_x86_64Linux2_1.5.7/bin/pythonsh

mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py
mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py
```

>We also have `prepare_flexreceptor4.py` which can be used to prepare a receptor with flexible side chains. However, this is not used in this tutorial.

To test if ADT is working add `pythonsh` to your `.bashrc` as an alias and run the following:
```bash
pythonsh mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py --help
```

### **2.2 Preparing receptor and ligand PDBQT files**
The receptor must be cleaned of water molecules and other non-residue molecules. The receptor must also be converted to a pdbqt file that includes charge (Q) and atom type (T) information. This can be done using the `prepare_receptor4.py` script from ADT.

There are two ways to prepare the receptor depending on the PDB file you have:

#### *2.2.A Receptor + ligand as PDB complex*
If the receptor and ligand already exist as a complex in a single PDB file, then run `prep_pdb.sh` with the following arguments. It will clean the file and split it into ligand and receptor pqbqt files.
```bash
prep_pdb.sh <pdb_file> <receptor_name>
```
```

#### *2.2.B Receptor on its own*

### Receptor