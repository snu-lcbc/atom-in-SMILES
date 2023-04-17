
from IPython.core.interactiveshell import default
from parameters import *
from utils import *
from transformer import *

from torch import nn
import torch

import sys, re
import argparse
import datetime
import copy
import heapq, json
from pprint import pprint
from pathlib import Path

from rdkit import Chem

import sentencepiece as spm
import numpy as np
import pandas as pd
import multiprocessing as mp

import atomInSmiles
import selfies as sf 
import deepsmiles

deepsmiles_converter = deepsmiles.Converter(rings=True, branches=True)

import logging

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

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__file__)


def setup(model, checkpoint_name):
    #assert os.path.exists(f"{ckpt_dir}/{checkpoint_name}"), f"There is no checkpoint named {checkpoint_name}."

    logger.info("Loading checkpoint...\n")
    # checkpoint = torch.load(checkpoint_name)
    checkpoint = torch.load(f"{ckpt_dir}/{checkpoint_name}")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optim_state_dict'])
    #best_loss = checkpoint['loss']

    return model


def custom_validation_fn(model, test_loader, model_type, method='greedy'):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{model_type}_src_sp.model")
    trg_sp.Load(f"{SP_DIR}/{model_type}_trg_sp.model")

    start_time = datetime.datetime.now()
    total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid = 0,0,0, 0,0,0,0,0, 0, 0, 0, 0,0, 0
    fp_tcs = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            #e_mask, d_mask = make_mask(src_input, trg_input)
            for j in  range(len(src_input)):
                # preparing src data for encoder
                src_j = src_input[j].unsqueeze(0).to(device) # (L) => (1, L)
                encoder_mask = (src_j != pad_id).unsqueeze(1).to(device) # (1, L) => (1, 1, L)
                # encoding src input
                src_j = model.src_embedding(src_j) # (1, L) => (1, L, d_model)
                src_j = model.positional_encoder(src_j) # (1, L, d_model)
                encoder_output = model.encoder(src_j, encoder_mask) # (1, L, d_model)
                s_src   = src_sp.decode_ids(src_input[j].tolist())
                s_truth = trg_sp.decode_ids(trg_output[j].tolist())
                if method == 'greedy':
                    s_pred = greedy_search(model, encoder_output, encoder_mask, trg_sp)
                    if model_type in ['char', 'atom', 'spe', ]:
                        if s_truth == s_pred:
                            exact += 1
                            tanimoto = 1.0
                        else:
                            tanimoto = FpSimilarity(s_truth.replace(' ', ''), s_pred.replace(' ', ''), radius=3, nBits=2048)
                    elif model_type in ['ais']:
                        decod_truth = AisDecoder(s_truth)
                        decod_pred = AisDecoder(s_pred)
                        if decod_truth == decod_pred:
                            exact +=1
                            tanimoto = 1.0
                        else:
                            #tanimoto = CalcAisTc(s_truth, s_pred)
                            tanimoto = FpSimilarity(decod_truth, decod_pred, radius=2, nBits=2048)
                    elif model_type in ['ae', 'ae0', 'ae2']:
                        tanimoto = FpSimilarity(AesDecoder(s_truth), AesDecoder(s_pred), radius=3, nBits=2048)
                        if s_truth == s_pred:
                            exact += 1
                    elif model_type =='kmer':
                        k_truth = DeKmer(s_truth)
                        k_pred = DeKmer(s_pred)
                        if k_truth == k_pred:
                            exact += 1
                            tanimoto = 1.0
                        else:
                            tanimoto = FpSimilarity(k_truth, k_pred, radius=3, nBits=2048)
                    elif model_type == 'ais2':
                        decod_truth = atomInSmiles.decode(s_truth)
                        decod_pred = atomInSmiles.decode(s_pred)
                        if decod_truth == decod_pred:
                            exact +=1
                            tanimoto = 1.0
                        else:
                            tanimoto = FpSimilarity(decod_truth, decod_pred, radius=2, nBits=2048)
                    elif model_type == 'selfies':
                        decod_truth = sf.decoder(s_truth.replace(' ', ''))
                        decod_pred = sf.decoder(s_pred.replace(' ', ''))
                        if decod_truth == decod_pred:
                            exact +=1
                            tanimoto = 1.0
                        else:
                            tanimoto = FpSimilarity(decod_truth, decod_pred, radius=2, nBits=2048)
                    elif model_type == 'deepsmi':
                        try:
                            decod_truth = deepsmiles_converter.decode(s_truth.replace(' ',''))
                            decod_pred = deepsmiles_converter.decode(s_pred.replace(' ', ''))
                        except:
                            decod_truth = s_truth 
                            decod_pred = s_pred
                        if decod_truth == decod_pred:
                            exact +=1
                            tanimoto = 1.0
                        else:
                            tanimoto = FpSimilarity(decod_truth, decod_pred, radius=2, nBits=2048)

                    preds = [s_pred,]

                elif method == 'beam':
                    s_preds, bscores = beam_search(model, encoder_output, encoder_mask, trg_sp)
                    max_tc = 0
                    best_pred=None
                    preds = list()
                    for s_pred in s_preds:
                        preds.append(s_pred)
                        if s_truth == s_pred:
                            exact +=1
                            max_tc = 1.0
                            best_pred = s_pred
                            break
                        elif model_type in ['char', 'atom', 'spe', ]:
                            if s_truth == s_pred:
                                exact += 1
                                max_tc = 1.0
                                best_pred = s_pred
                                break
                            else:
                                tanimoto = FpSimilarity(s_truth.replace(' ', ''), s_pred.replace(' ', ''), radius=3, nBits=2048)
                        elif model_type in ['ais']:
                            if AisDecoder(s_truth) == AisDecoder(s_pred):
                                exact += 1
                                max_tc = 1.0
                                best_pred = s_pred
                                break
                            else:
                                tanimoto = FpSimilarity(AisDecoder(s_truth), AisDecoder(s_pred), radius=3, nBits=2048)
                                #tanimoto = CalcAisTc(s_truth, s_pred)
                        elif model_type in ['ae', 'ae0', 'ae2']:
                            tanimoto = FpSimilarity(AesDecoder(s_truth), AesDecoder(s_pred), radius=3, nBits=2048)
                            if s_truth == s_pred:
                                exact += 1
                        elif model_type =='kmer':
                            k_truth = DeKmer(s_truth)
                            k_pred = DeKmer(s_pred)
                            if k_truth == k_pred:
                                exact += 1
                                max_tc = 1.0
                                best_pred = s_pred
                                break
                            else:
                                tanimoto = FpSimilarity(k_truth, k_pred, radius=3, nBits=2048)
                        if tanimoto > max_tc:
                            max_tc = tanimoto
                            best_pred = s_pred

                    tanimoto = max_tc
                    s_pred = best_pred
                    # --Beam ends

                fp_tcs.append(tanimoto)
                #predf.write(f"{aetc}|{fptc}|{s_truth}|{s_pred}\n")
                logger.info(f"Prediction|{tanimoto}|{s_truth}|{preds}|{s_src}")

                total += 1
                if tanimoto == 1.0:
                    _100 += 1
                elif tanimoto >= 0.90:
                    _90 += 1
                elif tanimoto >= 0.85:
                    _85 += 1
                elif tanimoto >= 0.80:
                    _80 += 1
                elif tanimoto >= 0.70:
                    _70 += 1
                elif tanimoto >= 0.60:
                    _60 += 1
                elif tanimoto >= 0.50:
                    _50 += 1
                elif tanimoto >= 0.40:
                    _40 += 1
                elif tanimoto >= 0.30:
                    _30 += 1
                elif tanimoto >= 0.20:
                    _20 += 1
                elif tanimoto > 0.0:
                    _00 += 1
                else:
                    _invalid += 1


    end_time = datetime.datetime.now()
    validation_time = end_time - start_time
    seconds = validation_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    elapsed_time = f"{hours}hrs {minutes}mins {seconds}secs"

    logger.info(f"{total, exact/total, _100/total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid} \t{elapsed_time=}")
    logger.info(f"{np.mean(fp_tcs) = }")
    logger.info(f"{total, exact/total, _100/total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid} \t{elapsed_time=}")
    logger.info(f"{np.mean(fp_tcs) = }")


def greedy_search(model, e_output, e_mask, trg_sp):
    last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.positional_encoder(trg_embedded)
        decoder_output = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        ) # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()

        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1

        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output


def beam_search(model, e_output, e_mask, trg_sp):
    cur_queue = PriorityQueue()
    #for k in range(beam_size):
    cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))

    finished_count = 0
    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            if pos == 0 and k > 0:
                continue
            else:
                node = cur_queue.get()

            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                ) # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        #new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        #if finished_count == beam_size:
        #    break

    #decoded_output = cur_queue.get().decoded
    #if decoded_output[-1] == eos_id:
    #    decoded_output = decoded_output[1:-1]
    #else:
    #    decoded_output = decoded_output[1:]
    #return trg_sp.decode_ids(decoded_output)
    all_candidates = list()
    scores  = [ ]
    for _ in range(beam_size):
        node = cur_queue.get()
        decoded_output = node.decoded
        scores.append(node.prob)
        all_candidates.append(trg_sp.decode_ids(decoded_output))

    return all_candidates, scores


def inference(model, input_sentence, model_type,  method):

    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{model_type}_src_sp.model")
    trg_sp.Load(f"{SP_DIR}/{model_type}_trg_sp.model")

    tokenized = src_sp.EncodeAsIds(input_sentence)
    logger.info(f"Indexed tokens: {tokenized}")
    src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

    start_time = datetime.datetime.now()

    logger.info("Model inference mode...")
    model.eval()
    src_data = model.src_embedding(src_data)
    src_data = model.positional_encoder(src_data)
    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if method == 'greedy':
        logger.info("Greedy decoding selected.")
        result = greedy_search(model, e_output, e_mask, trg_sp)
    elif method == 'beam':
        logger.info("Beam search selected.")
        result, _ = beam_search(model, e_output, e_mask, trg_sp)

    end_time = datetime.datetime.now()

    total_inference_time = end_time - start_time
    seconds = total_inference_time.seconds
    minutes = seconds // 60
    seconds = seconds % 60

    #print(f"Input: {input_sentence}")
    #print(f"Result: {result}")
    #print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
    return result


def main(args):
    if args.test_mode:
        test_loader = get_data_loader(args.model_type, TEST_NAME, args.batch_size, shuffle=False)

        if args.checkpoint_name:
            model = setup(build_model(model_type=args.model_type), args.checkpoint_name)
            _step = re.search(f'{args.model_type}_checkpoint_epoch_(.*).pth', args.checkpoint_name).group(1)
            # _step = re.search(f'{args.model_type}_checkpoint_epoch_(.*).pth', args.checkpoint_name.name).group(1)
            logger.info(f'PredictionStart| {_step}')
            custom_validation_fn(model, test_loader, args.model_type, method=args.decode)
            logger.info(f'PredictionEnd| {_step}')
        else:
            for ckpt in Path(ckpt_dir).iterdir():
                if ckpt.name.startswith(f'{args.model_type}_checkpoint_epoch_'):
                    print(ckpt)
                    _step = re.search(f'{args.model_type}_checkpoint_epoch_(.*).pth', ckpt.name).group(1)
                    model = setup(build_model(model_type=args.model_type), ckpt.name)
                    print('PredictionStart|', _step)
                    custom_validation_fn(model, test_loader, args.model_type, method=args.decode)
                    print('PredictionEnd|', _step)

    else:
        logger.info(f"Preprocessing input: {args.input}")
        model = setup(build_model(model_type=args.model_type), args.checkpoint_name)
        result = inference(model, args.input, args.model_type, args.decode)
        logger.info(f"Predicted: {result}")

    logger.info('Done!')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='model_type')
    parser.add_argument('--test_mode', action='store_true', help='Turn on testing mode')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--input', type=str, required='--test_mode' not in sys.argv, help='An input sequence')
    parser.add_argument('--decode', type=str, default='greedy', help="greedy or beam?")
    parser.add_argument('--beam_size', type=int, default=5, help="beam?")
    parser.add_argument('--checkpoint_name', type=str, default=None, help="checkpoint file name")
    parser.add_argument('--log', type=str, default=None, help="log file name")

    args = parser.parse_args()
    configure_logger(args.log)

    logger.info(args)
    assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."
    print(f'{args.decode} decoding searching method is selected.')
    if args.decode=='beam':
        logger.info('Beam size:', args.beam_size)
    beam_size = args.beam_size

    main(args)

