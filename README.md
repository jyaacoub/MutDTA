# MutDTA
Improving the precision oncology pipeline by providing binding affinity purtubations predictions on a pirori identified cancer driver genes.

# Current Progress
- [ ] Data preprocessing
  - [x] PDBbind - simple enough to use. Protein seq were downloaded from UniProt.
  - [x] PLATINUM - Working on getting mutated sequence data (done with mutalyzer)
  - [ ] KIBA and Davis - Kinase proteins are not super relevant but could be useful for pretraining since datasets are limited.
  - [ ] GENIE - This will require some physical docking methods since we have no binding affinity data for this.
- [ ] Docking baseline
  - [x] Set up Docking on cluster
  - [ ] Build scripts to automate ligand and protein prep (including grid for binding site).
- [ ] Model training
- [ ] Model evaluation

# Requirements
* AutoDock Tools script and pythonsh from MGL
* `.bashrc` with relevant aliases for molecule prep scripts like `prepare_ligand4.py`:

```bash
alias pythonsh='~/mgltools_x86_64Linux2_1.5.7/bin/pythonsh' # important
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
***
## 2. Installing Docking Tools (ADT and OpenBabel)

### **2.1 Install AutoDock Tools (ADT)**
For preparing the receptor and ligand we will use AutoDock Tools (ADT) from MGL. ADT is a GUI for preparing the receptor and ligand, however it comes with python scripts that we can run from the cmd line. Also it includes a prebuilt Python 2.0 interpreter called `pythonsh` that we can use to run the scripts.

Download ADT from http://mgltools.scripps.edu/downloads/ and unzip. 
```bash
wget https://ccsb.scripps.edu/mgltools/download/491/mgltools_x86_64Linux2_1.5.7p1.tar.gz
```
```bash
tar -xvzf mgltools_x86_64Linux2_1.5.7p1.tar.gz
```
Run install script
```bash
cd mgltools_x86_64Linux2_1.5.7 ; ./install.sh
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
### **2.1 Install OpenBabel**
Needs to be compiled from source (see: https://open-babel.readthedocs.io/en/latest/Installation/install.html#basic-build-procedure)
```bash
wget https://gigenet.dl.sourceforge.net/project/openbabel/openbabel/2.4.0/openbabel-openbabel-2-4-0.tar.gz
```
```bash
tar -xvzf openbabel-openbabel-2-4-0.tar.gz
```
Now we can compile and install it:
```bash
cd openbabel-openbabel-2-4-0; mkdir build; cd build
cmake .. -DPYTHON_BINDINGS=ON -DCMAKE_INSTALL_PREFIX=<LOCAL_PATH>
make # or make -j4 to use 4 cores
sudo make install
```
 Make sure to add to path by including `export PATH=<LOCAL_PATH>/bin/:$PATH` in `.bashrc`. 
>Note: For python bindings, we need to `sudo apt-get install libeigen3-dev` before running cmake.
>
>Also add `export PYTHONPATH=<LOCAL_PATH>:$PYTHONPATH` so that it can be imported in python.

Verify installation by running:
```bash
 obabel -H
```
>Note: `*/mgltools/mgltools_x86_64Linux2_1.5.7/bin/obabel` might interfere with this so remove it from path if present in `.bashrc`.

***
## 3. Preparing receptor and ligand PDBQT files
The receptor must be cleaned of water molecules and other non-residue molecules. The receptor must also be converted to a pdbqt file that includes charge (Q) and atom type (T) information. This can be done using the `prepare_receptor4.py` script from ADT.

## 3.1 Working with *PDBbind* dataset
There are many ways to prepare the receptor depending on the PDB file you have. For now we will focus on using the PDBbind dataset's layout to our advantage in prep.

Run the following to download PDBbind:
```bash
wget https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz
```
unzip without -v to avoid verbose output (takes a while):
```bash
tar -xzf PDBbind_v2020_refined.tar.gz
```

To prepare receptor and ligand we need to run `PDBbind_prepare.sh` with the following arguments:
```bash
PDBbind_prepare.sh <path> <ADT_path> [<shortlist>]
```
Where `<path>` is the path to the PDBbind dataset and `<ADT_path>` is the path to the ADT directory. Optionally we can also pass in `<shortlist>` the path to a csv file containing the PDB codes we want to prepare. This is useful if we want to prepare a subset of the PDBbind dataset.

Thats it! The script will automatically prepare the receptor and ligand and save them in the same location as the original ligand (sdf file) and receptor (pdb) files.


### **3.2**
The following are general instructions for preparing the receptor and ligand PDBQT files. This is useful if you have a PDB file that is not from PDBbind.
>#TODO: THIS IS NOT COMPLETE YET

### *3.2.A Receptor + ligand as PDB complex*
> For PDBbind we can download individual structures using wget as with `wget http://www.pdbbind.org.cn/v2007/10gs/10gs_complex.pdb` where `10gs` is the PDB ID

If the receptor and ligand already exist as a complex in a single PDB file we can automatically get the binding site information. For this, run `prep_pdbs.sh` with the following arguments. It will clean the files and split it into ligand (not usable) and receptor (usable) pqbqt files.

```bash
prep_pdbs.sh <path> <ADT_path>
```
>Make sure the PDB file has the following path format: `<path>/<pdbcode>/<pdbcode>.pdb`
>To get help message run `prep_pdb.sh` with no arguments.

The `<path>/<pdbcode>` directory should now contain the following files:
```bash
<pdbcode>.pdb
prep/<pdbcode>-split-<num_atoms>_ligand.pdbqt
prep/<pdbcode>-split-<num_atoms>_receptor.pdbqt
```

### *3.2.B Receptor on its own*
If the receptor is on its own in a PDB file we need to manually get the binding site information. We can still run `prep_pdbs.sh` but with `l` argument so that it only extracts the receptor.
```bash
prep_pdbs.sh <path> l <ADT_path>
```

### **3.3 Preparing Ligand PDBQT file**
For the ligand you need to download its SDF file and prepare it using OpenBabel or similar tools...
If you have the ligand name you can download it from PDB using the following address: `https://files.rcsb.org/ligands/download/{x}_ideal.sdf`

We can start from the SDF file and convert it to a PDBQT file using OpenBabel. To do so run the following:
```bash
obabel -isdf <path>/<ligand_name>_ideal.sdf -opdbqt -O <path>/<ligand_name>.pdbqt
```
or we can use the SMILES string to convert it to a PDBQT file using OpenBabel. To do so run the following:
```bash
obabel -:"<SMILES>" --gen3d -opdbqt -O <path>/<ligand_name>.pdbqt
```

### **3.4 Preparing Grid files**
For AutoDock Vina grid files and AutoGrid are not needed (see "AutoDock Tools Compatibility": https://vina.scripps.edu/manual/).

From Vina help message we can see how to input the search space:
```bash
Input:
  --receptor arg        rigid part of the receptor (PDBQT)
  --flex arg            flexible side chains, if any (PDBQT)
  --ligand arg          ligand (PDBQT)

Search space (required):
  --center_x arg        X coordinate of the center
  --center_y arg        Y coordinate of the center
  --center_z arg        Z coordinate of the center
  --size_x arg          size in the X dimension (Angstroms)
  --size_y arg          size in the Y dimension (Angstroms)
  --size_z arg          size in the Z dimension (Angstroms)

Output (optional):
  --out arg             output models (PDBQT), the default is chosen based on 
                        the ligand file name
  --log arg             optionally, write log file

Misc (optional):
  --cpu arg                 the number of CPUs to use (the default is to try to
                            detect the number of CPUs or, failing that, use 1)
  --seed arg                explicit random seed
  --exhaustiveness arg (=8) exhaustiveness of the global search (roughly 
                            proportional to time): 1+
  --num_modes arg (=9)      maximum number of binding modes to generate
  --energy_range arg (=3)   maximum energy difference between the best binding 
                            mode and the worst one displayed (kcal/mol)

Configuration file (optional):
  --config arg          the above options can be put here

Information (optional):
  --help                display usage summary
  --help_advanced       display usage summary with advanced options
  --version             display program version
```

This search space should be centered at the binding pocket and can be retrieved from the PDB file if provided as a complex. The size of the search space should be large enough to cover the entire binding pocket.

To create a config file with the search space provided, run the following:
```bash
python prep_conf.py -r <prep_path>
```
***
## 4. Running AutoDock Vina

Now that we have the prepared receptor, ligand, and `conf.txt` file set up, we can run AutoDock Vina.

To do so run the following:
```bash
vina --config <path>conf.txt
```

# 4.1 Running on PDBbind dataset
To run on the PDBbind dataset we can use the `run_vina.sh` script with the following arguments:
```bash
run_vina.sh <path> [<shortlist>]
```
Where `<path>` is the path to the PDBbind dataset. Optionally we can also pass in `<shortlist>` the path to a csv file containing the PDB codes we want to run. This is useful if we want to only dock a subset of the PDBbind dataset.

# Errors
See issues for errors.
Main issue rn is #1
