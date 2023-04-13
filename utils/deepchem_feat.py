# %load utils/deepchem_feat.py
import os
import re
import sys
import time
import pickle
import logging
import datetime
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

import rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles

import deepchem as dc
from deepchem.feat import Featurizer, DummyFeaturizer
from deepchem.data import Dataset, NumpyDataset, DiskDataset

# sys.path.append('/home/alatoo/projects/atom-in-SMILES/atomInSmiles')

logger = logging.getLogger(__name__)

current_time = datetime.datetime.now()
now = current_time.strftime("%Y-%m-%d-%H-%M")



class CustomFeaturizer(Featurizer):
    """
    Custom adaptation of deepchem's `Featurizer` for tokenization of chemical languages e.g SMILES, SELFIES etc.
    """

    def _featurize(self,) -> str:
        """Subclasses must be implemented """
        raise NotImplementedError()
    
    def featurize(self, datapoints: List[str], log_every_n: int = 1000, **kwargs) -> np.ndarray:
        if isinstance(datapoints, str) or isinstance(datapoints, Chem.rdchem.Mol):
            datapoints = [datapoints]
        if 'shard_size' in kwargs:
            shard_size = kwargs['shard_size']
        else:
            shard_size = self.shard_size
        if shard_size < len(datapoints):
            logger.info(f"Will be resharded: {len(datapoints)}/{shard_size}")
        featurizer_name = str(self)
        logger.info(f"Featurize: {featurizer_name}")
        features = []
        if isinstance(datapoints, DiskDataset):
            if 'save_dir' in kwargs:
                if os.path.exists(kwargs['save_dir']) and not kwargs.get('re_run', False):
                    logger.info(f"****Compiled dataset exists: Loading from {kwargs['save_dir']}")
                    loaded_dataset = DiskDataset(kwargs['save_dir'])
                    if loaded_dataset.get_shard_size() > shard_size:
                        loaded_dataset.reshard(shard_size)
                    return loaded_dataset
                
            logger.info(f"{type(datapoints) = } \t {datapoints}")
            ids, y, w = [], [], []
            for i, datapoint in enumerate(datapoints.X):
                if i % log_every_n == 0:
                    logger.info(f"Featurizing datapoint: {i}")
                try:
                    mol = Chem.MolFromSmiles(datapoint)
                    if mol is not None:
                        datapoint = Chem.MolToSmiles(mol)
                        encoded = self._featurize(datapoint) 
                        encoded_list = encoded.split()
                        features.append(encoded)
                        ids.append(datapoints.ids[i])
                        y.append(datapoints.y[i])
                        w.append(datapoints.w[i])
                        self.corpus.extend(encoded_list)
                        len_ = len(encoded_list)
                        if len_ > self.max_len:
                            self.max_len = len_
                except Exception as e:
                     logging.warning(f"Faild to process datapoint at {i}, {datapoint}; Exception: {e}")
        else:
            if isinstance(datapoints, list):
                logger.debug(f'datapoint: {len(datapoints)=}; example: {datapoints[0]}; {type(datapoints[0])}')
            for i, datapoint in enumerate(datapoints):
                if i  % log_every_n == 0:
                    logger.info(f"Featurizing datapoint: {i}")
                try:
                    mol = Chem.MolFromSmiles(datapoint)
                    if mol is not None:
                        datapoint = Chem.MolToSmiles(mol)
                        encoded = self._featurize(datapoint) 
                        encoded_list = encoded.split()
                        features.append(encoded)
                        self.corpus.extend(encoded_list)
                        len_ = len(encoded_list)
                        if len_ > self.max_len:
                            self.max_len = len_
                except Exception as e:
                    logging.warning(f"Faild to process datapoint at {i}, {datapoint}; Exception: {e}")
                
        
        # return np.asarray(features)
        token_count = Counter(self.corpus)
        self.vocab = { i:j for i, (j, c) in enumerate(sorted(token_count.items(), key=lambda x: x[1], reverse=True))}
        self.vocab_size = len(self.vocab) 
        logger.debug(f"Featurized: {features[0]}")
        logger.info(f"# unique tokens: {self.vocab_size}")
        if self.return_as == 'binary':
            logger.info(f"Binarizing the datapoints... It will return as binary vectors")
            if isinstance(datapoints, DiskDataset):
                one_hot_encoded = [self.getOneHot(item) for item in features]
                return DiskDataset.create_dataset([
                    (np.asarray(one_hot_encoded), np.asarray(y), np.asarray(w), np.asarray(ids))
                ], tasks=datapoints.tasks, data_dir=datapoints.save_dir)
            else: 
                one_hot_encoded = [self.getOneHot(item) for item in features]
                return np.asarray(one_hot_encoded)
        
        elif self.return_as == 'tokenized':
            logger.info(f"Tokenizing the datapoints... It will return as indices")
            self.vocab = { 
                token:idx
                for idx, (token, c) in enumerate(
                    sorted(token_count.items(), key=lambda x: x[1], reverse=True),2)
            }
            self.vocab['<pad>'] = 0
            self.vocab['<unk>'] = 1
            self.vocab_size = len(self.vocab)
            logger.info(f"Vocab size: {self.vocab_size} (<pad>, <unk> are included) and max length: {self.max_len}")
            self.tokenized = [self.getTokenIdx(item) for item in features]
            if isinstance(datapoints, DiskDataset):
                return DiskDataset.create_dataset([
                    (np.asarray(self.tokenized), np.asarray(y), np.asarray(w), np.asarray(ids))
                ], tasks=datapoints.tasks, data_dir=datapoints.save_dir)
            else:
                return np.asarray(self.tokenized)
        else:
            if isinstance(datapoints, DiskDataset):
                return DiskDataset.create_dataset([
                    (np.asarray(features), np.asarray(y), np.asarray(w), np.asarray(ids))
                ], tasks=datapoints.tasks, )
            else:
                return np.asarray(features)
            
        
    def get_binary(self, datapoints: DiskDataset,
                   save_dir,
                   shard_size=5000,
                   tasks=None,
                   re_run=False,
                  ) -> DiskDataset:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created a new dir: {save_dir}")
        else:
            if os.path.exists(os.path.join(save_dir, 'metadata.csv.gzip')) and not re_run:
                logger.info(f"Compiled dataset exists. Loading from {save_dir}")
                dataset = DiskDataset(save_dir)
                if dataset.get_shard_size() != shard_size:
                    logger.info(f'Resharding...{shard_size}')
                    dataset.reshard(shard_size)
                
                return dataset

        time1 = time.time()
        if not self.corpus:
            corpus = []
            for X, y, w, ids in datapoints.itersamples():
                corpus.extend(X.split())

            token_count = Counter(corpus)
            self.vocab = { token:i for i, (token, count) in enumerate(sorted(token_count.items(), key=lambda x: x[1], reverse=True))}.copy()
            self.vocab_size = len(self.vocab)
            del corpus

        metadata_rows = []
        for shard_num, (X, y, w, ids) in enumerate(datapoints.itershards()):
            X_binary = []
            if shard_num == 0:
                if tasks is None and y is not None:
                    # The line here assumes that y generated by shard_generator is a numpy array
                    tasks = np.array([0]) if y.ndim < 2 else np.arange(
                        y.shape[1])
                    
            # Initialize a matrix of zeros with shape (num_data, vocab_size)
            # binary_matrix = csr_matrix((num_data, vocab_size), dtype=np.int8)

            # # Loop over each sentence and set the appropriate indices to 1
            # for i, sentence in enumerate(nbbbp_df['ais']):
            #     words = sentence.split()
            #     word_indices = [vocab[word] for word in words]
            #     binary_matrix[i, word_indices] = 1

            # for x in X:
            #     xbin = self.getOneHot(x)
            #     X_binary.append(xbin)
            X_binary = csr_matrix((len(X), self.vocab_size), dtype=np.int8)
            for i, x in enumerate(X):
                tokens = x.split()
                token_idx = [ vocab[token] for token in tokens]
                X_binary[i, token_idx] = 1
                
            base_name = f"shard-{shard_num}"
            metadata_rows.append(
                DiskDataset.write_data_to_disk(save_dir, base_name, X_binary, y, w, ids,)
                # DiskDataset.write_data_to_disk(save_dir, base_name, np.asarray(X_binary), y, w, ids,)
            )
            logger.info(f"Processed: {shard_num=}")

        metadata_df = DiskDataset._construct_metadata(metadata_rows)
        DiskDataset._save_metadata(metadata_df, save_dir, tasks)

        time2 = time.time()
        logger.info(f"TIMING: dataset construction took {(time2-time1):.3f} s")
        return DiskDataset(save_dir)

    def get_binary_(self, datapoints: DiskDataset,
                   save_dir,
                   shard_size=5000,
                   tasks=None,
                   re_run=False,
                  ) -> DiskDataset:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created a new dir: {save_dir}")
        else:
            if os.path.exists(os.path.join(save_dir, 'metadata.csv.gzip')) and not re_run:
                logger.info(f"Compiled dataset exists. Loading from {save_dir}")
                dataset = DiskDataset(save_dir)
                if dataset.get_shard_size() != shard_size:
                    logger.info(f'Resharding...{shard_size}')
                    dataset.reshard(shard_size)
                
                return dataset

        time1 = time.time()
        if not self.corpus:
            corpus = []
            for X, y, w, ids in datapoints.itersamples():
                corpus.extend(X.split())

            token_count = Counter(corpus)
            self.vocab = { i:j for i, (j, c) in enumerate(sorted(token_count.items(), key=lambda x: x[1], reverse=True))}.copy()
            self.vocab_size = len(self.vocab)
            del corpus

        metadata_rows = []
        for shard_num, (X, y, w, ids) in enumerate(datapoints.itershards()):
            X_binary = []
            if shard_num == 0:
                if tasks is None and y is not None:
                    # The line here assumes that y generated by shard_generator is a numpy array
                    tasks = np.array([0]) if y.ndim < 2 else np.arange(
                        y.shape[1])
            for x in X:
                xbin = self.getOneHot(x)
                X_binary.append(xbin)
            base_name = f"shard-{shard_num}"
            metadata_rows.append(
                DiskDataset.write_data_to_disk(save_dir, base_name, np.asarray(X_binary), y, w, ids,)
            )
            logger.info(f"Processed: {shard_num=}")

        metadata_df = DiskDataset._construct_metadata(metadata_rows)
        DiskDataset._save_metadata(metadata_df, save_dir, tasks)

        time2 = time.time()
        logger.info(f"TIMING: dataset construction took {(time2-time1):.3f} s")
        return DiskDataset(save_dir)


    def getOneHot(self, item):
        one_hot = np.zeros(self.vocab_size)
        items = item.split()
        idx = [i for i,k in self.vocab.items() if k in items]
        one_hot[idx] = 1
        return one_hot
    
    def getTokenIdx(self, tokenized_text):
        token_ids = [ self.vocab.get(token, 1) for token in tokenized_text.split()]
        left = self.max_len - len(token_ids)
        padding = [self.vocab['<pad>']] * left
        return token_ids + padding
        
    def smiles_tokenizer(self, smiles):
        """
        Tokenize a SMILES molecule or reaction
        """
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        # assert smiles == ''.join(tokens)
        return ' '.join(tokens)
    
    def decode(self, token_ids:np.array) -> str:
        assert self.vocab is not None, f"Vocabulary is not build yet. Please run the featurizer beforhand"
        pass
        

    
class AisFeaturizer(CustomFeaturizer):
    def __init__(self, return_as=None, shard_size=5000):
        super(AisFeaturizer, self).__init__()
        try:
            import atomInSmiles
        except Except as e:
            raise (f"Couldn't import atomInSmiles: {e}")
        self.tokenizer  = atomInSmiles.encode
        
        self.return_as = return_as
        self.shard_size = shard_size
        self.corpus = []
        self.max_len = 0
        self.vocab_size = 0
        
    def _featurize(self, smiles:str) -> str:
        try:
            tokenized = self.tokenizer(smiles)
            if tokenized is not None or tokenized != '':
                return tokenized
            else:
                return 0
        except:
            return 0
        

class SmilesFeaturizer(CustomFeaturizer):
    def __init__(self, return_as=None, shard_size=5000):
        super(SmilesFeaturizer, self).__init__()
        import re
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(pattern)
        self.return_as = return_as
        self.shard_size = shard_size
        self.corpus = []
        self.max_len = 0
        self.vocab_size = 0
        
    def _featurize(self, smiles:str) -> str:
        """
        Tokenize a SMILES molecule or reaction
        """
        tokens = [token for token in self.regex.findall(smiles)]
        assert smiles == ''.join(tokens)
        try:
            tokenized = ' '.join(tokens)#, len(tokens)
            if tokenized is not None or tokenized != '':
                return tokenized
            else:
                return 0
        except:
            return 0
    

class SmilesPEFeaturizer(CustomFeaturizer):
    def __init__(self, vocab_file='./data/SPE_ChEMBL.txt', return_as=None, shard_size=5000):
        super(SmilesPEFeaturizer, self).__init__()
        try:
            import codecs
            from SmilesPE.tokenizer import SPE_Tokenizer
        except Except as e:
            raise (f"Couldn't import SmilesPE: {e}")
            
        spe_vob= codecs.open(vocab_file)
        spe = SPE_Tokenizer(spe_vob)
        self.tokenizer = spe.tokenize
        self.return_as = return_as
        self.shard_size = shard_size
        self.corpus = []
        self.max_len = 0
        self.vocab_size = 0
        
    def _featurize(self, smiles:str) -> str:
        try:
            tokenized = self.tokenizer(smiles)
            if tokenized is not None or tokenized != '':
                return tokenized
            else:
                return 0
        except:
            return 0
        


class SelfiesFeaturizer(CustomFeaturizer):
    def __init__(self, return_as=None, shard_size=5000):
        super(SelfiesFeaturizer, self).__init__()
        try:
            import selfies as sf
        except Except as e:
            raise (f"Couldn't import selfeis: {e}")
        self.sf = sf
        self.tokenizer  = sf.encoder 
        self.return_as = return_as
        self.shard_size = shard_size
        self.corpus = []
        self.max_len = 0
        self.vocab_size = 0
        
    def _featurize(self, smiles:str) -> str:
        sf_encoded = list(self.sf.split_selfies(self.tokenizer(smiles)))
        try:
            tokenized = ' '.join(sf_encoded)#, len(tokens)
            if tokenized is not None or tokenized != '':
                return tokenized
            else:
                return 0
        except:
            return 0


class DeepSmilesFeaturizer(CustomFeaturizer):
    def __init__(self, return_as=None, shard_size=5000):
        super(DeepSmilesFeaturizer, self).__init__()
        try:
            import deepsmiles
        except Except as e:
            raise (f"Couldn't import deepchem: {e}")
        converter = deepsmiles.Converter(rings=True, branches=True)
        self.tokenizer = converter.encode
        self.return_as = return_as
        self.shard_size = shard_size
        self.corpus = []
        self.max_len = 0
        self.vocab_size = 0

    def _featurize(self, smiles:str) -> str:
        try:
            tokenized = deepsmi = self.tokenizer(smiles)
            if tokenized is not None or tokenized != '':
                return self.smiles_tokenizer(deepsmi)
            else:
                return 0
        except:
            return 0
        


def smi_canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return 0
    except:
        return 0
