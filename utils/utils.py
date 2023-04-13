import re
import os
import sys 
import time 
import logging
import argparse
from pathlib import Path 

import numpy as np
import pandas as pd

from typing import List, Tuple

import deepchem as dc
from deepchem.feat import Featurizer
from deepchem.data import DiskDataset


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, CanonSmiles 
# from rdkit.Chem import rdMolStandardize


def configure_logger(log_file=None, level=logging.INFO):
    logging.basicConfig(
        level=level,
        # use module name as output format
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # don't use the default date format
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ],
    )
logger = logging.getLogger(__name__)

def smiles_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return ' '.join(tokens)



def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

