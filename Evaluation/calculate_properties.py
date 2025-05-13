"""
Calculate a set of outcomes for a list of SMILES, writing the complete
distribution and not just a summary statistic.
"""

import argparse
import os
import numpy as np
import pandas as pd
import sys
from itertools import chain
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem.GraphDescriptors import BertzCT
from tqdm import tqdm

# suppress Chem.MolFromSmiles error output
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
# import from rdkit.Contrib module
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
import npscorer

# set working directory


# import functions
from functions import clean_mols, read_smiles

### CLI
parser = argparse.ArgumentParser(
        description='Calculate a series of properties for a set of SMILES')
parser.add_argument('--smiles_file', type=str,
                    help='file containing SMILES')
parser.add_argument('--reference_file', type=str,
                    help='file containing a reference set of SMILES')
parser.add_argument('--output_dir', type=str,
                    help='directory to save output to')
parser.add_argument('--selfies', dest='selfies',
                    help='calculate outcomes for molecules in SELFIES format',
                    action='store_true')
parser.add_argument('--deepsmiles', dest='deepsmiles',
                    help='calculate outcomes for molecules in DeepSMILES format',
                    action='store_true')
parser.add_argument('--stop_if_exists', dest='stop_if_exists',
                    action='store_true')
parser.set_defaults(stop_if_exists=False)
args = parser.parse_args()


# optionally stop if output file already exists


output_file = (r'D:\PFAS生成\PFAS_properties_output_part3_20241108.csv')


# create results container
res = pd.DataFrame()

# read SMILES and convert to molecules
smiles = read_smiles('output_part3.txt')
mols = [mol for mol in clean_mols(smiles, selfies=args.selfies,
                                  deepsmiles=args.deepsmiles) if mol]
canonical = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in mols]

# also read the reference file
ref_smiles = read_smiles(r"D:\PFAS生成\train_PFAS.txt")
ref_mols = [mol for mol in clean_mols(ref_smiles) if mol]
ref_canonical = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in \
                 ref_mols]

## drop known molecules
canonical = [sm for sm in canonical if sm not in ref_canonical]
# re-parse molecules
mols = [mol for mol in clean_mols(canonical) if mol]

# calculate descriptors
## heteroatom distribution
elements = [[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in mols]
counts = np.unique(list(chain(*elements)), return_counts=True)
## molecular weights
mws = [Descriptors.MolWt(mol) for mol in mols]
## logP
logp = [Descriptors.MolLogP(mol) for mol in tqdm(mols)]
## Bertz TC
tcs = [BertzCT(mol) for mol in tqdm(mols)]
## TPSA
tpsa = [Descriptors.TPSA(mol) for mol in mols]
## QED
qed = []
for mol in tqdm(mols):
    try:
        qed.append(Descriptors.qed(mol))
    except OverflowError:
        pass

## % of sp3 carbons
pct_sp3 = [Lipinski.FractionCSP3(mol) for mol in tqdm(mols)]
## % heteroatoms
pct_hetero = [Lipinski.NumHeteroatoms(mol) / mol.GetNumAtoms() for mol in \
              tqdm(mols)]
## number of rings
rings = [Lipinski.RingCount(mol) for mol in tqdm(mols)]
## SA score
SA = []
for mol in tqdm(mols):
    try:
        SA.append(sascorer.calculateScore(mol))
    except (OverflowError, ZeroDivisionError):
        pass

## NP-likeness
fscore = npscorer.readNPModel()
NP = [npscorer.scoreMol(mol, fscore) for mol in tqdm(mols)]


# add all outcomes to data frame
res = pd.DataFrame({'SMILES':canonical,'Molecular weight':mws,'LogP': logp,'BertzTC':tcs,'TPSA':tpsa,'QED':qed,
                    '% sp3 carbons':pct_sp3,'% heteroatoms':pct_hetero,'# of rings':rings,'Synthetic accessibility score':SA,
                    'Natural product-likeness score':NP})
'''
for idx, element in enumerate(counts[0]):
    atom_count = counts[1][idx]
    res = res.append(pd.DataFrame({'outcome': '# atoms, ' + element,
                                   'value': [atom_count] }))'''
res.to_csv(output_file)

# write output
#res.to_csv(output_file, index=False, compression='gzip')
