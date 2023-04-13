import warnings 
import numpy as np

from rdkit import Chem # enough for basics
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdMolDescriptors import GetAtomPairFingerprint
from rdkit.Chem.rdMolDescriptors import GetTopologicalTorsionFingerprint
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP


#predefined

def MACCS(mol, return_bits=False, **kwargs):
    fp = GetMACCSKeysFingerprint(mol)
    return list(fp.GetOnBits()) if return_bits else fp


def Avalon(mol, return_bits=False, **kwargs):
    fp = GetAvalonFP(mol, nBits=512)
    return list(fp.GetOnBits()) if return_bits else fp

#path based

def RDK4(mol,  return_bits=False, **kwargs):
    fp = RDKFingerprint(mol, minPath=2, maxPath=4, nBitsPerHash=1)
    return list(fp.GetOnBits()) if return_bits else fp

def RDK4_default(mol,  return_bits=False, **kwargs):
    fp = RDKFingerprint(mol)
    return list(fp.GetOnBits()) if return_bits else fp

def RDK4_L(mol, return_bits=False, **kwargs):
    fp = RDKFingerprint(mol, minPath=2, maxPath=4, nBitsPerHash=1, branchedPaths=False)
    return list(fp.GetOnBits()) if return_bits else fp


def HashAP(mol, return_bits=False, **kwargs):
    fp = GetHashedAtomPairFingerprint(mol, minLength=1, maxLength=6)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def TT(mol, return_bits=False, **kwargs):
    fp = GetTopologicalTorsionFingerprint(mol, **kwargs)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def HashTT(mol, return_bits=False, **kwargs):
    fp = GetHashedTopologicalTorsionFingerprint(mol, **kwargs)
    return list(fp.GetNonzeroElements()) if return_bits else fp


#circular env

def AEs(mol, return_bits=False, **kwargs):
    fp = GetMorganFingerprint(mol, radius=1)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def ECFP0(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=0, nBits=2048)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def ECFP2(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=1, nBits=2048)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def ECFP2_(mol, return_bits=False, **kwargs):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=2048)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    bits = array.nonzero()
    return bits[0].tolist() if return_bits else fp


def ECFP4(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=2, nBits=2048)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def ECFP4_(mol, return_bits=False, **kwargs):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    bits = array.nonzero()
    return bits[0].tolist() if return_bits else fp


def FCFP2(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=1, nBits=2048, useFeatures=True)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def FCFP4(mol, return_bits=False, **kwargs):
    fp = GetHashedMorganFingerprint(mol, radius=2, nBits=2048, useFeatures=True)
    return list(fp.GetNonzeroElements()) if return_bits else fp


def FpSimilarity(smiles1, smiles2,
                 metric=DataStructs.TanimotoSimilarity, #DiceSimilarity
                 fingerprint=rdMolDescriptors.GetMorganFingerprint,
                 rdLogger=False, # RDKit logger
                 **kwargs,):
    RDLogger.EnableLog('rdApp.*') if rdLogger else RDLogger.DisableLog('rdApp.*')
    try:
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=True)
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=True)
        if mol2 is not None and mol1 is not None:
            fp1 = fingerprint(mol1, **kwargs)
            fp2 = fingerprint(mol2, **kwargs)
            return metric(fp1, fp2)
        else:
            if rdLogger:
                warnings.warn(f'{smiles1=}, {smiles2=}')
            return 0
    except:
        return 0
