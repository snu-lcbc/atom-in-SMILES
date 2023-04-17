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
#src_vocab_size = {'char':52, 'atom' : 68, 'ais' :658,   'spe':2852, 'kmer': 11253, 'ae2' : 913331, 'ae0' : 143, 'ae_smi2':2602, 'ae_t0':2395,}
#trg_vocab_size = {'char':53, 'atom' : 86, 'ais' :843,   'spe':2859, 'kmer': 12297, 'ae2' : 913582, 'ae0' : 202, 'ae_smi2':2650, 'ae_t0':2791,}
src_vocab_size = {'char':42, 'atom' : 68, 'ae_smi2' : 3988, 'ae2' : 8416, 'ais' : 658, 'ae0' : 109, 'spe':2852, 'kmer': 11253, 'ais2':658, 'deepsmi': 82, 'selfies':92 }
trg_vocab_size = {'char':50, 'atom' : 86, 'ae_smi2' : 4452, 'ae2' : 8828, 'ais' : 843, 'ae0' : 129, 'spe':2859, 'kmer': 12297, 'ais2':843, 'deepsmi': 103, 'selfies':110 }
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
