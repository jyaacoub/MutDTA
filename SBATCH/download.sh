#!/bin/bash
cd ~/projects/def-sushant/jyaacoub/MutDTA/
source .venv/bin/activate

python -u << EOF

from src.data_prep import Downloader 

Downloader.download_SDFs(
    ligand_ids=
    [
        'CHEMBL1200978',
        'CHEMBL1201182',
        'CHEMBL1201438',
        'CHEMBL1201561',
        'CHEMBL1201576',
        'CHEMBL1201827',
        'CHEMBL159',
        'CHEMBL1908360',
        'CHEMBL2105741',
        'CHEMBL2360464',
        'CHEMBL3137309',
        'CHEMBL359744',
        'CHEMBL3989514',
        'CHEMBL413',
        'CHEMBL428647',
        'CHEMBL90555',
        'CHEMBL92'
    ],
    save_dir= './downloaded_ligands',
    max_workers=5
)

EOF