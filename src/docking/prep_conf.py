"""
This creates the configuration file with a focus on the binding site.

Using ligand position to identify where the binding region is.

# NOTE: Binding info is provided during split_pdb step instead 
# this way we dont have to keep ligand pdb since it is not good for docking.
"""

import argparse, os
from helpers.format_pdb import get_df

parser = argparse.ArgumentParser(description='Prepares config file for AutoDock Vina.')
parser.add_argument('-p', metavar='--prep_path', type=str,
                    help="Directory containing prepared ligand and protein. \
                    With file names ending in 'ligand.pdqt' or 'receptor.pdqt'.", required=False)

parser.add_argument('-r', metavar='--receptor', type=str, 
                    help='Path to pdbqt file containing sole protein.', required=False)
parser.add_argument('-l', metavar='--ligand', type=str, 
                    help='Path to pdbqt file containing sole ligand.', required=False)

parser.add_argument('-pp', metavar='--pocket_path', type=str,
                    help='binding pocket pdb file from PDBbind', required=False)

parser.add_argument('-o', metavar='--output', type=str,
                    help='Output config file path. Default is to save it \
                    in the same location as the receptor as "conf.txt"', required=False)
parser.add_argument('-c', metavar='--conf_path', type=str,
                    help='Path to config file to use as template. Default is to use \
                        AutoDock Vina defaults (see: https://vina.scripps.edu/manual/#config).', 
                        required=False)

if __name__ == '__main__':
    args = parser.parse_args()

    # (args.p is None) implies (args.r is not None and args.l is not None)
    # !p -> (r&l)
    # p || (r&l)
    if not((args.p is not None) or (args.r is not None and args.l is not None)):
        parser.error('Either prep_path or receptor and ligand must be provided.')
        
    if args.p is not None:
        # Automatically finding protein and ligand files
        # should be named *_receptor.pdbqt and *_ligand.pdbqt if created by split_pdb.py
        for file in os.listdir(args.p):
            if file.endswith('receptor.pdbqt'):
                args.r = f'{args.p}/{file}'
            elif file.endswith('ligand.pdbqt'):
                args.l = f'{args.p}/{file}'
        
    if args.c is None:
        # These are the default values set by AutoDock Vina (see: https://vina.scripps.edu/manual/#config)
        # placing them here for reference
        conf = {
            "energy_range": 3,   # maximum energy difference between the best binding mode and the worst one (kcal/mol)
            "exhaustiveness": 8, # exhaustiveness of the global search (roughly proportional to time)
            "num_modes": 9,      # maximum number of binding modes to generate
            #"cpu": 1,           # num cpus to use. Default is to automatically detect.
        }
    conf["receptor"] = args.r
    conf["ligand"] = args.l
    
    # saving binding site info if path provided
    if args.pp is not None:
        pocket_df = get_df(open(args.pp, 'r').readlines())
        conf["center_x"] = pocket_df["x"].mean()
        conf["center_y"] = pocket_df["y"].mean()
        conf["center_z"] = pocket_df["z"].mean()
        conf["size_x"] = pocket_df["x"].max() - pocket_df["x"].min()
        conf["size_y"] = pocket_df["y"].max() - pocket_df["y"].min()
        conf["size_z"] = pocket_df["z"].max() - pocket_df["z"].min()
        
    # FIXME: Error at line 67 (check h4h history for replication details)
    """
    H4H@~/projects/MutDTA/src/docking/bash_scripts/PDBbind$ python ../../prep_conf.py -r ~/data/refined-set/4k7i/4k7i_protein.pdbqt -l ~/data/refined-set/4k7i/4k7i_ligand.pdbqt -pp  ~/data/refined-set/4k7i/4k7i_pocket.pdb -o  ~/data/refined-set/4k7i/4k7i_conf.txt
Traceback (most recent call last):
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 1630, in _ensure_numeric
    x = float(x)
ValueError: could not convert string to float: 'Nov12.00211.12411.91811.02110.09011.36912.1509.90011.32110.4799.0158.11311.16412.60212.6568.7729.5627.4127.2097.9486.1165.5935.6155.3304.8914.3445.5845.9715.3446.1975.9203.8603.0032.9412.2632.1641.4481.3887.2167.4358.0359.2459.5958.5149.3298.8027.3689.89410.76912.02912.27911.0819.8399.49812.79712.56313.96813.62314.31212.54411.95612.14611.63411.83911.07811.45110.97810.93110.31511.21511.0219.1218.1798.58812.15412.27313.03214.34514.88213.32412.05312.25112.87312.81313.33011.82013.24714.83214.26916.11615.98216.34217.09417.27517.90718.44515.48815.05015.57314.75515.33515.22215.92915.55316.97016.36317.23813.42912.96812.60012.95113.07911.10010.11610.1978.68712.50711.99312.90312.25011.24012.47413.06512.31914.54510.03910.2329.3117.8227.1309.91411.41411.96211.7265.3686.1655.1834.5953.8706.4477.5176.9918.5184.8825.4694.4173.2202.5225.5356.6006.2607.9487.2398.9318.5878.2508.7568.4299.72710.4918.4817.2245.9924.8374.1974.5675.3116.1175.0863.5682.9913.3579.9619.22211.18412.37712.27911.04210.36910.11511.19315.77116.08416.15617.27118.29616.62016.69017.14816.55117.11715.35714.56414.87113.62914.93315.53513.99619.75320.53618.38517.47716.56417.69018.51316.81717.57516.94916.06817.03716.57615.13718.89119.36219.71119.84119.25220.63121.24920.67322.10522.71320.03618.56318.23617.54814.86012.729-0.8789.3149.814'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 1634, in _ensure_numeric
    x = complex(x)
ValueError: complex() arg is a malformed string

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/cluster/home/t122995uhn/projects/MutDTA/src/docking/bash_scripts/PDBbind/../../prep_conf.py", line 67, in <module>
    conf["center_x"] = pocket_df["x"].mean()
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/generic.py", line 11847, in mean
    return NDFrame.mean(self, axis, skipna, level, numeric_only, **kwargs)
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/generic.py", line 11401, in mean
    return self._stat_function(
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/generic.py", line 11353, in _stat_function
    return self._reduce(
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/series.py", line 4816, in _reduce
    return op(delegate, skipna=skipna, **kwds)
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 93, in _f
    return f(*args, **kwargs)
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 155, in f
    result = alt(values, axis=axis, skipna=skipna, **kwds)
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 418, in new_func
    result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 706, in nanmean
    the_sum = _ensure_numeric(values.sum(axis, dtype=dtype_sum))
  File "/cluster/tools/software/centos7/python/3.10.9/lib/python3.10/site-packages/pandas/core/nanops.py", line 1637, in _ensure_numeric
    raise TypeError(f"Could not convert {x} to numeric") from err
TypeError: Could not convert Nov12.00211.12411.91811.02110.09011.36912.1509.90011.32110.4799.0158.11311.16412.60212.6568.7729.5627.4127.2097.9486.1165.5935.6155.3304.8914.3445.5845.9715.3446.1975.9203.8603.0032.9412.2632.1641.4481.3887.2167.4358.0359.2459.5958.5149.3298.8027.3689.89410.76912.02912.27911.0819.8399.49812.79712.56313.96813.62314.31212.54411.95612.14611.63411.83911.07811.45110.97810.93110.31511.21511.0219.1218.1798.58812.15412.27313.03214.34514.88213.32412.05312.25112.87312.81313.33011.82013.24714.83214.26916.11615.98216.34217.09417.27517.90718.44515.48815.05015.57314.75515.33515.22215.92915.55316.97016.36317.23813.42912.96812.60012.95113.07911.10010.11610.1978.68712.50711.99312.90312.25011.24012.47413.06512.31914.54510.03910.2329.3117.8227.1309.91411.41411.96211.7265.3686.1655.1834.5953.8706.4477.5176.9918.5184.8825.4694.4173.2202.5225.5356.6006.2607.9487.2398.9318.5878.2508.7568.4299.72710.4918.4817.2245.9924.8374.1974.5675.3116.1175.0863.5682.9913.3579.9619.22211.18412.37712.27911.04210.36910.11511.19315.77116.08416.15617.27118.29616.62016.69017.14816.55117.11715.35714.56414.87113.62914.93315.53513.99619.75320.53618.38517.47716.56417.69018.51316.81717.57516.94916.06817.03716.57615.13718.89119.36219.71119.84119.25220.63121.24920.67322.10522.71320.03618.56318.23617.54814.86012.729-0.8789.3149.814 to numeric
    """
    

    # saving config file
    if args.o is None:
        args.o = '/'.join(conf["receptor"].split('/')[:-1]) + '/conf.txt'
        
    with open(args.o, 'a') as f:
        for key, value in conf.items():
            f.write(f'{key} = {value}\n')
        
        # adding custom config file if provided
        if args.c is not None:
            with open(args.c, 'r') as c:
                for line in c:
                    # making sure no duplicates are added
                    if line.split(' = ')[0] not in conf.keys():
                        f.write(line)