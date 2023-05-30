[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![J. Cheminformatics DOI](https://img.shields.io/badge/J._Cheminformatics-10.1186%2Fs13321--023--00725--9-blue)](https://doi.org/10.1186/s13321-023-00725-9)


## Atom-in-SMILES tokenization.

 >Ucak UV, Ashyrmamatov I, Lee J (2023) Improving the quality of chemical language model outcomes with atom-in-SMILES tokenization. J Cheminformatics 15:55. https://doi.org/10.1186/s13321-023-00725-9
  
Tokenization is an important preprocessing step in natural language processing
that may have a significant influence on prediction quality. This research showed
that the traditional SMILES tokenization has a certain limitation that results in
tokens failing to reflect the true nature of molecules. To address this issue, we
developed the atom-in-SMILES tokenization scheme that eliminates ambiguities
in the generic nature of SMILES tokens. Our results in multiple chemical
translation and molecular property prediction tasks demonstrate that proper
tokenization has a significant impact on prediction quality. In terms of prediction
accuracy and token degeneration, atom-in-SMILES is more effective method in
generating higher-quality SMILES sequences from AI-based chemical models
compared to other tokenization and representation schemes. We investigated the
degrees of token degeneration of various schemes and analyzed their adverse
effects on prediction quality. Additionally, token-level repetitions were quantified,
and generated examples were incorporated for qualitative examination. We
believe that the atom-in-SMILES tokenization has a great potential to be
adopted by broad related scientific communities, as it provides chemically
accurate, tailor-made tokens for molecular property prediction, chemical
translation, and molecular generative models.

[Tutorial - Google Colab Notes](https://colab.research.google.com/drive/1Kry-nleryw5f6azttWPa7I8lCKLQwxzT?usp=sharing)

<hr style="background: transparent; border: 0.2px dashed;"/>

## Installation
It can be installed using pip.
```shell
pip install atomInSmiles
```
or clone it from the GitHub repository and install locally. 
```shell
git clone https://github.com/snu-lcbc/atom-in-SMILES
cd atom-in-SMILES
python setup.py install
```
   
## Usage & Demo
 Brief descriptions of the main functions: 
| Function                              | Description                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| ``atomInSmiles.encode``               | Converts a SMILES string into Atom-in-SMILES tokens. |
| ``atomInSmiles.decode``               | Converts an Atom-in-SMILES tokens into SMILES string. |
| ``atomInSmiles.similarity``           | Calcuates Tanimoto coefficient of two Atom-inSMILSE tokens. |

```python
import atomInSmiles

smiles = 'NCC(=O)O'

# SMILES -> atom-in-SMILES 
ais_tokens = atomInSmiles.encode(smiles) # '[NH2;!R;C] [CH2;!R;CN] [C;!R;COO] ( = [O;!R;C] ) [OH;!R;C]'

# atom-in-SMILES -> SMILES
decoded_smiles = atomInSmiles.decode(ais_tokens) #'NCC(=O)O'

assert smiles == decoded_smiles

```
**NOTE:** By default, it first canonicalizes the input SMILES. In order to get atom-in-Smiles tokens with the same order of SMILES, the input SMILES should be provided with atom map numbers.

```python
from rdkit.Chem import MolFromSmiles, MolToSmiles
import atomInSmiles

import atomInSmiles
# ensuring the order of SMILES in atom-in-SMILES. 
smiles = 'NCC(=O)O'
mol = MolFromSmiles(smiles)
random_smiles = MolToSmiles(mol, doRandom=True) # e.g 'C(C(=O)O)N' 

# mapping atomID into SMILES srting
tmp = MolFromSmiles(random_smiles)
for atom in tmp.GetAtoms():
    atom.SetAtomMapNum(atom.GetIdx())
smiles_1 = MolToSmiles(tmp) # 'C([C:1](=[O:2])[OH:3])[NH2:4]' 

# SMILES -> atom-in-SMILES
ais_tokens_1 = atomInSmiles.encode(smiles_1, with_atomMap=True) # '[CH2;!R;CN] ( [C;!R;COO] ( = [O;!R;C] ) [OH;!R;C] ) [NH2;!R;C]'

# atom-in-SMILES -> SMILES
decoded_smiles_1 = atomInSmiles.decode(ais_tokens_1) # 'C(C(=O)O)N'

assert random_smiles == decoded_smiles_1
```
   
<hr style="background: transparent; border: 0.5px dashed;"/>

## Implementations & Results

| Implementation | Items |Description| 
|----------------|------:|----|
| Single-step retrosynthesis | `python src/predict.py` |  to conduct an inference with the trained model|
|  | `--model_type` |  (`SMILES`, `SELFIES`, `DeepSmiles`, `SmilesPE`, `AIS`)
|  | `--checkpoint_name` |  name of the checkpoint file [checkpoints files](https://drive.google.com/file/d/1tDKIKrKWevgTgJjF8QZpd1IKxZr_Pc1q/view?usp=sharing)
|  | `--input` |  Tokenized input sequence
| Molecular Property Prediction | [Molecular-property-prediction.ipynb](https://github.com/snu-lcbc/atom-in-SMILES/blob/main/Molecular-property-prediction.ipynb)| __MoleculeNet__: Classification (ESOL, FreeSolv, Lipo.), Regression (BBBP, BACE, HIV)|
|  Normalized repetition rate | [Normalized-Repetition-Rates.ipynb](https://github.com/snu-lcbc/atom-in-SMILES/blob/main/Normilized-Repetition-Rates.ipynb)| Natural products, drugs, metal complexes, lipids, stereoids, isomers |
| Fingerprint nature of AIS | [AIS-as-fingerprint.ipynb](https://github.com/snu-lcbc/atom-in-SMILES/blob/main/AIS-as-fingerprint.ipynb) | AIS fingerprint resolution|
| Single-token repetition (rep-l) | [rep-l_USPTO50k.ipynb](https://github.com/snu-lcbc/atom-in-SMILES/blob/main/rep-l_USPTO50k.ipynb) | __USPTO-50K__, retrosynthetic translations|
| input-output equivalent mapping | [GDB13-results.ipynb](https://github.com/snu-lcbc/atom-in-SMILES/blob/main/GDB13-results.ipynb) | Augmented subset of __GDB-13__, noncanon-2-canon translations|

For example, in retrosynthesis task:
```python
python src/predict.py --model_type AIS  --checkpoint_name AIS_checkpoint.pth
 --input='[CH3;!R;O] [O;!R;CC] [C;!R;COO] ( = [O;!R;C] ) [c;R;CCS] 1 [cH;R;CC] [c;R;CCC] ( [CH2;!R;CC] [CH2;!R; CC] [CH2;!R;CC] [c;R;CCN] 2 [cH;R;CC] [c;R;CCC] 3 [c;R;CNO] ( = [O;!R;C] ) [nH;R;CC] [c;R;NNN] ( [NH2 ;!R;C] ) [n;R;CC] [c;R;CNN] 3 [nH;R;CC] 2 ) [cH;R;CS] [s;R;CC] 1'
```

### Cite this work
```bash
@article{10.1186/s13321-023-00725-9, 
year = {2023}, 
title = {{Improving the quality of chemical language model outcomes with atom-in-SMILES tokenization}}, 
author = {Ucak, Umit V. and Ashyrmamatov, Islambek and Lee, Juyong}, 
journal = {Journal of Cheminformatics}, 
doi = {10.1186/s13321-023-00725-9}, 
pages = {55}, 
number = {1}, 
volume = {15}, 
keywords = {}
}
```

### License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
