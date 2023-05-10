#%% testing download of sequences from FASTA files
import requests as r
from data_processing.pdbbind import get_prot_seq, save_prot_seq, excel_to_csv, prep_save_data
import pandas as pd
import os, re

