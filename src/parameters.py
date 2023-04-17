import torch

# Path or parameters for data
DATA_DIR = 'data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0 ; sos_id = 1 ; eos_id = 2 ; unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
src_vocab_size = {'AIS':658, 'SMILES' : 68, 'SmilesPE':2852, 'DeepSMILES': 82,  'SELFIES':92 }
trg_vocab_size = {'AIS':843, 'SMILES' : 86, 'SmilesPE':2859, 'DeepSMILES': 103, 'SELFIES':110 }
character_coverage = 1.0
sp_model_type = 'word'

# Parameters for Transformer & training
learning_rate = 0.00
batch_size = 100
seq_len = 150
num_heads = 8
num_layers = 6
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.2
num_epochs = 1000
beam_size = 5
ckpt_dir = 'saved_models'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
