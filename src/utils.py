from torch.utils.data import Dataset, DataLoader
from parameters import *
from transformer import *

import torch
import sentencepiece as spm
import numpy as np
import pandas as pd
import heapq
import warnings
import re
from pathlib import Path

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs



def build_model(model_type):
    print(f"{model_type} molecular model is building...")
    #print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{SP_DIR}/{model_type}_src_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{SP_DIR}/{model_type}_trg_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word

    #print(f"The size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")

    return Transformer(src_vocab_size=len(src_i2w), trg_vocab_size=len(trg_i2w)).to(device)


def make_mask(src_input, trg_input):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask

# custom data structure for beam search method
class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")

class PriorityQueue():

    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)


#################
# Data loaders

def get_data_loader(model_type, file_name, batch_size=100, shuffle=True):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{model_type}_src_sp.model")
    trg_sp.Load(f"{SP_DIR}/{model_type}_trg_sp.model")

    print(f"Getting source/target {file_name} for {model_type} molecular...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{model_type}_{file_name}", 'r', encoding="utf-8") as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{model_type}_{file_name}", 'r', encoding="utf-8") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list, src_sp) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list, trg_sp) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list, src_sp):
    tokenized_list = []
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list

def process_trg(text_list, trg_sp):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in text_list:
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list

class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]


# Metrics for evaluation
def FpSimilarity(smiles1, smiles2,
                 metric=DataStructs.TanimotoSimilarity, #DiceSimilarity
                 fingerprint=Chem.rdMolDescriptors.GetHashedMorganFingerprint,
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

def CalcAesTc(s_truth, s_pred):
    lpred0 = smiles_tokenizer(s_pred)
    ltruth0 = smiles_tokenizer(s_truth)
    lpred = []
    for i in set(lpred0):
        if '[' == i[0]:
            lpred.append(i)

    ltruth = []
    for i in set(ltruth0):
        if '[' == i[0]:
            ltruth.append(i)
    return len(set(ltruth) & set(lpred)) / float(len(set(ltruth) | set(lpred)))


def CalcAisTc(s_truth, s_pred):
    lpred = []
    for i in set(s_pred.split()):
        if '[' == i[0]:
            lpred.append(i)

    ltruth = []
    for i in set(s_truth.split()):
        if '[' == i[0]:
            ltruth.append(i)
    return len(set(ltruth) & set(lpred)) / float(len(set(ltruth) | set(lpred)))


def CalcFpTc(truth_smi, pred_smi, metric=Chem.DataStructs.TanimotoSimilarity):
    RDLogger.DisableLog('rdApp.*')
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi, sanitize=True)
        if pred_mol is None:
            return 0
        truth_mol = Chem.MolFromSmiles(truth_smi, sanitize=True)
        if truth_mol is None:
            return 0
    except:
        return 0
#     return Chem.DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(truth_mol), Chem.RDKFingerprint(pred_mol), metric=metric)
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(truth_mol,2,nBits=2048), AllChem.GetMorganFingerprintAsBitVect(pred_mol,2,nBits=2048))


def getSubstructSmi(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        ####
        if '[' in atom.GetSmarts():
            symbol = atom.GetSmarts()
        else:
            sym = atom.GetSmarts()
            symbol = '['+sym
            if nHs:
                symbol += 'H'
                if nHs>1:
                    symbol += '%d'%nHs
            if isInRing:
                symbol += ';R'
            else:
                symbol += ';!R'
            symbol += ';D%d'%deg
            symbol += "]"
#         print(f"{atom.GetSmarts() = } {sym = }")
        symbols.append(symbol)
    smi = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
    smi2 = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    return smi,smi2


def AesEncoder(smiles, radius):
    import re
    atomID_sma = {}
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atomID_sma[atom.GetIdx()] = getSubstructSmi(mol, atom.GetIdx(), radius)[1]   # by [1] selecting SMARTS

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens_list = [token for token in regex.findall(Chem.MolToSmiles(mol))]
#     print(tokens_list)
    assert Chem.MolToSmiles(mol) == ''.join(tokens_list)
    atomID = 0
    updateTokens = []
    for i, token in enumerate(tokens_list):
        if i ==0:
            updateTokens.append(atomID_sma[atomID])
            atomID+=1
        else:
            try:
                symbol, idx = token.split(':')
                updateTokens.append(atomID_sma[atomID])
                atomID+=1
            except:
                updateTokens.append(token)
#     return " ".join(updateTokens), len(updateTokens)
    return " ".join(updateTokens)


def AesDecoder(new_tokens_list):
    RDLogger.DisableLog('rdApp.*')
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smarts = ''
    for new_token in new_tokens_list.split():
        symbols = [token for token in regex.findall(new_token)]
        try:
            smarts += symbols[0]
        except:
            pass
    try:
        mol_sma = Chem.MolFromSmarts(smarts)
        if mol_sma != None:
            return Chem.MolToSmiles(mol_sma)
    except:
        return None


def getAis(mol,atomID,radius):

    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        # assert len(atomsToUse) == len(set(atomsToUse)), atomsToUse
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        neighbors = atom.GetNeighbors()
        isAromatic = atom.GetIsAromatic()
        chirality = atom.GetChiralTag()
        symbol = '['
        symbol += atom.GetSmarts()
        if '@' not in symbol:
            if nHs:
                symbol += 'H'
                if nHs>1:
                    symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        for i in neighbors:
            symbol += f';{i.GetSymbol()}'
        symbol += "]"
        symbols.append(symbol)
    return Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)


def AisEncoder(smiles, radius=0):

    import re
    smiles_list = smiles.split('.')
    aes = []
    for smiles in smiles_list:
        atomID_sma = {}
        mol = Chem.MolFromSmiles(smiles)
        tmp_radius = radius
        while (2*tmp_radius) >= mol.GetNumAtoms() and tmp_radius != 0:
            tmp_radius -=1

        for atom in mol.GetAtoms():
            atomID_sma[atom.GetIdx()] = getAis(mol, atom.GetIdx(), tmp_radius)#[1] # by [1] selecting SMARTS

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens_list = [token for token in regex.findall(Chem.MolToSmiles(mol))]
        assert Chem.MolToSmiles(mol) == ''.join(tokens_list), f"{Chem.MolToSmiles(mol)} \n {''.join(tokens_list)}"
        atomID = 0
        updateTokens = []
        for i, token in enumerate(tokens_list):
            if i ==0:
                updateTokens.append(atomID_sma[atomID])
                atomID+=1
            else:
                try:
                    symbol, idx = token.split(':')
                    updateTokens.append(atomID_sma[atomID])
                    atomID+=1
                except:
                    updateTokens.append(token)
        aes += updateTokens +["."]
    return  " ".join(aes[:-1]).strip()


def AisDecoder(new_tokens_list, rdLogger=False):

    RDLogger.EnableLog('rdApp.*') if rdLogger else RDLogger.DisableLog('rdApp.*')
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smarts = ''
    for new_token in new_tokens_list.split():
        try:
            token =regex.findall(new_token)[0]
        except:
            token =''
        if '[[' in token:
            smarts += token[1:]
        else:
            if '[' in token:
                sym = token[1:-1].split(';')
                if "H" in sym[0]:
                    smarts += regex.findall(sym[0])[0]
                else:
                    smarts += sym[0]
            else:
                smarts += token
    try:
        mol_sma = Chem.MolFromSmarts(smarts, mergeHs=True)
        if mol_sma != None:
            return Chem.CanonSmiles(Chem.MolToSmiles(mol_sma))
    except:
        return None



def DeKmer(kmer):
    smiles = ' '
    list_kmer = kmer.split()
    for i in range(len(list_kmer)):
        if len(list_kmer) - 1 ==1:
            smiles += list_kmer[i]
        else:
            smiles += list_kmer[i][0]
    return smiles.strip()


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def smiles_tokenizer(smi):
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    #assert smi == ''.join(tokens)
    #return ' '.join(tokens), len(tokens)
    return tokens




# old metrics
#################
# Preprocessing of input SMILES

def getSmarts(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        symbol = '['+atom.GetSmarts()
        if nHs:
            symbol += 'H'
            if nHs>1:
                symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        symbol += ';D%d'%deg
        symbol += "]"
        symbols.append(symbol)
    try:
        smart = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    except (ValueError, RuntimeError) as ve:
        print('atom to use error or precondition bond error')
        return
    return smart


def getAtomEnvs(smiles, radii=[0, 1], radius=1, nbits=1024):
    """
    A function to extract atom environments from the molecular SMILES.

    Parameters
    ----------
    smiles: str
        Molecular SMILES
    radii: list
        list of radii you would like to obtain atom envs.
    radius: int
        radius of MorganFingerprint
    nbits: int
        size of bit vector for MorganFingerprint

    Returns
    -------
    tuple
        a list of atom envs and a string type of this list
    """

    assert max(radii) <= radius, f"the maximum of radii should be equal or lower than radius, but got {max(radius)}"

    molP = Chem.MolFromSmiles(smiles.strip())
    if molP is None:
        #warnings.warn(f"There is a semantic error in {smiles}")
        raise Exception (f"There is a semantic error in {smiles}")

    sanitFail = Chem.SanitizeMol(molP, catchErrors=True)
    if sanitFail:
        raise Exception (f"Couldn't sanitize: {smiles}")

    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molP,radius=radius, nBits=nbits, bitInfo=info)# condition can change

    info_temp = []
    for bitId,atoms in info.items():
        exampleAtom,exampleRadius = atoms[0]
        description = getSmarts(molP,exampleAtom,exampleRadius)
        info_temp.append((bitId, exampleRadius, description))

    #collect the desired output in another list
    updateInfoTemp = []
    for k,j in enumerate(info_temp):
        if j[1] in radii:                           # condition can change
            updateInfoTemp.append(j)
        else:
            continue

    tokens_str = ''
    tokens_list = []
    for k,j in enumerate(updateInfoTemp):
        tokens_str += str(updateInfoTemp[k][2]) + ' ' #[2]-> selecting SMARTS description
        tokens_list.append(str(updateInfoTemp[k][2]))  # condition can change

    return tokens_list, tokens_str.strip()



def molGen(input):
    Slist = list()
    input.strip()
    if ' . ' in input:
        input = input.split(' . ')
        R1 = input[0].strip().split()
        R2 = input[1].strip().split()
        Slist.append(R1)
        Slist.append(R2)
    else:
        R = input.split()
        Slist.append(R)
    return Slist

def tanimoto(truth, prediction, i, j):
    return len(set(truth[i]) & set(prediction[j])) / float(len(set(truth[i]) | set(prediction[j])))

def timing(f):
    import time
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        seconds = time2 - time1
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        print(f'{f.__name__} -> Elapsed time: {hours}hrs {minutes}mins {seconds:.3}secs')

        return ret
    return wrap


def db_search(query, dbdir, topk):
    query_set = set(query.strip().split())

    resultq = []
    for file in Path(dbdir).iterdir():
        if file.name.endswith('smarts'):
            with open(file, 'r') as fp:
                for i, item in enumerate(fp.readlines(), 1):
                    if i % 1000000 == 0:
                        print(i)
                    sequence = item.strip().split('\t')
                    smiles = sequence[0].strip()
                    aes_str = sequence[1].strip()
                    aes_set = set(sequence[1].strip().split())
                    tanimoto = tc(query_set, aes_set)
                    if tanimoto >= 0.8:
                        #result.append((location, query_noform, smile, nbit_noform, tanimoto))
                        heapq.heappush(resultq, (-tanimoto, query, aes_str, smiles))
    c = 1
    candidates = []
    until = topk if len(resultq) > topk else len(resultq)

    while c <= until:
        c += 1
        try:
            candidates.append(heapq.heappop(resultq))
        except:
            pass
    return candidates


@timing
def mp_dbSearch(results_dict, dbdir, topk=5):
    import multiprocessing as mp
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count()+2)
    jobs = []
    for k, v in results_dict.items():
        job = pool.apply_async(db_search, (v, dbdir, topk ))
        jobs.append((k, job))

    results = []
    for k, job in jobs:
        #results.append(job.get())
        candidates = job.get()
        for tanimoto, query, aes_str, smiles in candidates:
            results.append([k, query, aes_str, smiles, -tanimoto])

    #return results
    return pd.DataFrame(results, columns=['Tree', 'Model_Prediction', "DB_AEs", "DB_SMILES", "DB_Tc"])



def tc(query, nbit):
    a = len(query)
    b = len(nbit)
    c = len(set(query).intersection(nbit))
    if c != 0:
        return c / (a + b - c)
    else:
        return 0


def similarity(truth, prediction):
# Sdict = Similarity dictiontionary, Nlist = NameList, Vlist = Value list
    Sdict = dict()
    if len(truth) == 2 and len(prediction) == 2:
        # ground truth A >> B + C. Prediction A >> D + E
        Nlist = ['DB', 'DC', 'EB', 'EC']
        Vlist = [(0,0), (1,0), (0,1), (1,1)]

        for i in range(4):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DB'] >= Sdict['DC']:
            del Sdict['DC']
        else:
            del Sdict['DB']

        if Sdict['EB'] >= Sdict['EC']:
            del Sdict['EC']
        else:
            del Sdict['EB']

    # Condition 2

    elif len(truth) == 1 and len(prediction) == 2:
        # ground truth A >> G. Prediction A >> D + E
        Nlist = ['DG', 'EG']
        Vlist = [(0,0), (0,1)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['DG'] >= Sdict['EG']:
            del Sdict['EG']
        else:
            del Sdict['DG']

    # Condition 3

    elif len(truth) == 2 and len(prediction) == 1:
        # ground truth A >> B + C. Prediction A >> F
        Nlist = ['FB', 'FC']
        Vlist = [(0,0), (1,0)]

        for i in range(2):
            Sdict[Nlist[i]] = tanimoto(truth, prediction, Vlist[i][0], Vlist[i][1])

        if Sdict['FB'] >= Sdict['FC']:
            del Sdict['FC']
        else:
            del Sdict['FB']

    # Condition 4

    elif len(truth) == 1 and len(prediction) == 1:
        # ground truth A >> G. Prediction A >> F
        Nlist = ['FG']
        Vlist = [(0,0)]
        Sdict[Nlist[0]] = tanimoto(truth, prediction, Vlist[0][0], Vlist[0][1])

    else:
        Sdict['Prediction'] = 0

    return Sdict

