# AutoDock Vina Procedure

## Requirements
* Python 3.10 and pip libraries in `requirements.txt`
* AutoDock Tools script and pythonsh from MGL
* `.bashrc` with relevant aliases for molecule prep scripts like `prepare_ligand4.py`:

```bash
alias pythonsh='~/mgltools_x86_64Linux2_1.5.7/bin/pythonsh' # important
```
***
## 1. Installing AutoDock Vina and tools
### **1.1 AutoDock Vina**
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
### **1.2 Install AutoDock Tools (ADT)**
For preparing the receptor and ligand we will use AutoDock Tools (ADT) from MGL. ADT is a GUI for preparing the receptor and ligand, however it comes with python scripts that we can run from the cmd line. Also it includes a prebuilt Python 2.0 interpreter called `pythonsh` that we can use to run the scripts.

Download ADT from http://mgltools.scripps.edu/downloads/ and unzip. 
```bash
wget https://ccsb.scripps.edu/mgltools/download/491/mgltools_x86_64Linux2_1.5.7p1.tar.gz
```
For v1.5.6 use https://ccsb.scripps.edu/mgltools/download/495/mgltools_x86_64Linux2_1.5.6p1.tar.gz
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
### **1.3 Install OpenBabel**
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

## 2. Download PDBbind refined-set
There are many ways to prepare the receptor depending on the PDB file you have. For this tutorial we will be using PDBbind refined-set (v2020) which is a set of 4,951 protein-ligand complexes with high-quality experimental binding data.

Run the following to download PDBbind:
```bash
wget https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz
```
unzip without -v to avoid verbose output (takes a while):
```bash
tar -xzf PDBbind_v2020_refined.tar.gz
```

***

## 3. Preparing receptor and ligand PDBQT files
The receptor must be cleaned of water molecules and other non-residue molecules. The receptor must also be converted to a pdbqt file that includes charge (Q) and atom type (T) information. This can be done using the `prepare_receptor4.py` script from ADT.

For this I have created a bash script `PDBbind_prepare.sh` that will prepare the receptor and ligand for all the PDBbind complexes. For a help message run `PDBbind_prepare.sh` with no arguments.
```
Usage: src/docking/bash_scripts/PDBbind_prepare.sh path ADT template [OPTIONS]
       path     - path to PDBbind dir containing pdb for protein to convert to pdbqt.
       ADT - path to MGL root  (e.g.: '~/mgltools_x86_64Linux2_1.5.7/')
       template - path to conf template file (create empty file if you want vina defaults).
Options:
       -sl --shortlist: path to csv file containing a list of pdbcodes to process.
              Doesn't matter what the file is as long as the first column contains the pdbcodes.
       -cd --config-dir: path to store new configurations in.
              Default is to store it with the prepared receptor as <PDBCode>_conf.txt
```
>note template is a file containing the configuration for vina. If you want to use the default vina configuration, create an empty file and pass it as the template. See [Vina docs](https://vina.scripps.edu/manual/#config) for examples.

Thats it! The script will automatically prepare the receptor and ligand and save them in the same location as the original ligand (sdf file) and receptor (pdb) files.

## 4. Running Vina
To run vina use script `src/docking/bash_scripts/run_vina.sh`. For a help message run `run_vina.sh` with no arguments.
```
Usage: ./src/docking/bash_scripts/run_vina.sh <path> <shortlist>
         path - path to PDBbind dir containing pdb for protein to convert to pdbqt.
         shortlist - path to csv file containing a list of pdbcodes to process.
                    Doesnt matter what the file is as long as the first column contains the pdbcodes.
         conf_dir (optional) - path to configuration dir for each pdb. Default is the same path as pdb file itself.
```

## 5. Data analysis
To Extract binding affinity predictions made by vina run `src/docking/python_helpers/extract_vina_out.py`. For help run `python extract_vina_out.py -h`.
```
usage: extract_vina_out.py [-h] [-sl --shortlist] [-fr] [-dm] [-fm] path out_csv

Extracts vina results from out files. (Example use: python extract_vina_out.py ./data/refined-set/ data/vina_out/run3.csv -sl ./data/shortlists/no_err_50/sample.csv)

positional arguments:
  path             Path to directory with PDBbind file structure or simple dir containing just vina outs (see arg -fr).
  out_csv          Output path for the csv file (e.g.: PATH/TO/FILE/vina_out.csv)

options:
  -h, --help       show this help message and exit
  -sl --shortlist  Shortlist csv file containing pdbcodes to extract. Otherwise extracts all in PDBbind dir. (REQUIRED IF -fr IS SET)
  -fr              (From Run) - Extract vina out data from a 'run' directory containing all vina outs in a single dir. This will also set -dm to true.
  -dm              (Dont Move) - Don't move vina_log and vina_out files to new dir with same name as out_csv file.
  -fm              (Force Move) - Force move vina_log and vina_out files. Overrides -fr default action on -dm.
```


## Errors
See [issues](/../../issues) for errors.


# SBATCH procedure
For those wishing to run this on a cluster, I have also created some sbatch scripts to run the above commands. Here we are starting from a PDBbind directory format with protein pdb and ligand sdf files.

The following steps are labeled **A** if you are running this for the first time, and **B** if this is the second time you are running this (i.e. you have already prepared the PDBQT files).

## 0. Setup for Multiprocessing
If you dont want to parallelize just make sure to modify the SLURM `--array` option in the `multi_*.sh` scripts to be a single array.
### 0.1 Split csv
Create a directory splitting a shortlist file into partitions to run docking over using `src/docking/bash_scripts/split_csv.sh`. Shortlist file can be any file as long as the first column contains the pdbcodes you are interested in docking. For help run `split_csv.sh` with no arguments.
```
Usage: ./split_csv.sh [-h] input_file output_dir num_partitions
Split a CSV file into multiple partitions named <part#>.csv.

Arguments:
  -i               Ignore the header row in the input file (optional).
  input_file       Path to the input CSV file.
  output_dir       Directory to store the output partitions.
  num_partitions   Number of partitions to create.

Example:
  ./split_csv.sh -i input.csv output_partitions 5
```
### 0.2 Modify *'multi_*'* sbatch scripts
Before continuing, remember to modify the `multi_*` sbatch scripts to point to the correct paths. `'#NOTE'` has been added to the lines that need to be modified/checked (except for path definitions).

## 1.A Prepare PDBQT and conf files
Run the following script `src/docking/sbatch/multi_prepare.sh`. This will run `src/docking/bash_scripts/PDBbind_prepare.sh` on all the PDBbind complexes in the shortlist file.

## 1.B Prepare just conf files
Assuming we have PDBQT files prepared we can just create conf files by running `src/docking/sbatch/multi_prep_conf.sh`. Which will run `src/docking/bash_scripts/prep_conf_only.sh` on all the PDBbind complexes in the shortlist file. For help run `prep_conf_only.sh` with no arguments.
```
Usage: ./src/docking/bash_scripts/prep_conf_only.sh <path> <template> <shortlist> [<config_dir>]
         path      - path to PDBbind dir containing pdb for protein to convert to pdbqt (ABSOLUTE PATH).
         template  - path to conf template file (create empty file if you want vina defaults).
         shortlist - path to csv file containing a list of pdbcodes to process.
                     Doesnt matter what the file is as long as the first column contains the pdbcodes.
Options:
         config_dir (optional) - path to store new configurations in. default is to store it with the protein as {PDBCode}_conf.txt
```
## 2. Run Vina
### 2.2A run multidock
Using `src/docking/sbatch/multi_dock.sh` we can run multiple docking runs at the same time to speed up the process. Make sure to adjust # of processes to match # of partitions. This will run `src/docking/bash_scripts/run_vina.sh` on all the PDBbind complexes in the shortlist file. For help run `run_vina.sh` with no arguments.
```
Usage: ./src/docking/bash_scripts/run_vina.sh <path> <shortlist>
         path - path to PDBbind dir containing pdb for protein to convert to pdbqt.
         shortlist - path to csv file containing a list of pdbcodes to process.
                    Doesnt matter what the file is as long as the first column contains the pdbcodes.
         conf_dir (optional) - path to configuration dir for each pdb. Default is the same path as pdb file itself.
```

### 2.2B run multidock from conf
Same as **2.2A** but change `conf_dir` in `src/docking/sbatch/multi_dock.sh` to match input conf files directory.

## 3. Data analysis
Follow the same steps as in [**5. Data analysis**](#5-data-analysis) above.
