[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Atom-in-SMILES tokenization.
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


<hr style="background: transparent; border: 0.2px dashed;"/>

## Installation
It can be installed directly from the GitHub repository.
```shell
pip install git+https://github.com/snu-lcbc/atom-in-SMILES.git
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

## Implementations
### Single-step retrosynthesis
First, [checkpoints files](https://drive.google.com/file/d/1tDKIKrKWevgTgJjF8QZpd1IKxZr_Pc1q/view?usp=sharing) should be downloaded and extracted.

Run below commands to conduct an inference with the trained model.
`python src/predict.py` 
* `--model_type`: Specifies the type of model to use. `AIS` for Atom-in-SMILES tokenization scheme (`SMILES`, `SELFIES`, `DeepSmiles`, `SmilesPE`).
* `--checkpoint_name`: Specifies the name of the checkpoint file that contains the trained model parameters.
* `--input`: Specifies the input tokenized sequence for which the prediction should be made.
* `--decode`: {greedy,beam}   Decoding method  (default: greedy)
* `--beam_size`: BEAM_SIZE Beam size (a number of candidates for RetroTRAE) (default: 3)    
For example:
```python
python src/predict.py --model_type AIS  --checkpoint_name AIS_checkpoint.pth
 --input='[CH3;!R;O] [O;!R;CC] [C;!R;COO] ( = [O;!R;C] ) [c;R;CCS] 1 [cH;R;CC] [c;R;CCC] ( [CH2;!R;CC] [CH2;!R; CC] [CH2;!R;CC] [c;R;CCN] 2 [cH;R;CC] [c;R;CCC] 3 [c;R;CNO] ( = [O;!R;C] ) [nH;R;CC] [c;R;NNN] ( [NH2 ;!R;C] ) [n;R;CC] [c;R;CNN] 3 [nH;R;CC] 2 ) [cH;R;CS] [s;R;CC] 1'
```

## Cite
[![DOI](https://zenodo.org/ggh)

### License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
