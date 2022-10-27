[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Atom-in-SMILES tokenization.
Tokenization is an important preprocessing step in natural language processing that may have a significant influence on prediction quality. 
Atom-in-SMILES concept is developed to resolve the ambiguities in the genericness of SMILES tokens.
Considering the prediction accuracy and token degeneration comparisons, atom-in-SMILES appears as an effective method to draw higher quality SMILES sequences out of AI-based chemical models than other tokenization schemes. For more details and results, please refer to the following reference ----.

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
