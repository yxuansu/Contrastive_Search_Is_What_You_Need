# coding=utf-8
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def parse_data_path(data_path_prefix, translation_direction, shot):
    data_path_prefix = data_path_prefix + '/{}/'.format(translation_direction)
    if shot == 0:
        train_path = None
    else:
        train_path = data_path_prefix + 'train.json'
    test_path = data_path_prefix + 'test.json'
    print ('Train path is {}'.format(train_path))
    print ('Test path is {}'.format(test_path))
    return train_path, test_path

def inference_one_instance(args, data, index, model, tokenizer, bos_token_id, eos_token_id, 
    cuda_available, device):
    input_ids = [bos_token_id] + data.context_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    _, prefix_len = input_ids.size()
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_method, decoding_len = args.decoding_method, args.decoding_len
    with torch.no_grad():
        if decoding_method == 'beam':
            beam_width = 4
            output  = model.beam_search(input_ids=input_ids, decoding_len=decoding_len,
                beam_width = beam_width, end_of_sequence_token_id = eos_token_id, early_stop = True) 
        elif decoding_method == 'nucleus':
            output  = model.nucleus_sampling(input_ids=input_ids, nucleus_p=0.95, 
                decoding_len=decoding_len, end_of_sequence_token_id = eos_token_id, early_stop = True) 
        elif decoding_method == 'contrastive':
            k, alpha = 3, 0.4
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=k, alpha=alpha, 
                decoding_len=decoding_len, end_of_sequence_token_id = eos_token_id, early_stop = True, 
                block_context_degeneration_penalty=False) 
        else:
            raise Exception('Wrong Decoding Method!')

    output_text = tokenizer.decode(output[prefix_len:])
    prediction = output_text.strip()

    # parse output result
    context = data.context_list[index]
    reference = data.reference_list[index].strip()
    one_res_dict = {
        'context': context,
        'reference': reference,
        'prediction': prediction
    }
    return one_res_dict

def parse_prediction_result(in_f):
    with open(in_f) as f:
        data = json.load(f)

    prediction_list, reference_list = [], []
    for item in data:
        one_prediction = item['prediction'].strip()
        one_reference = item['reference'].strip()
        prediction_list.append(one_prediction)
        reference_list.append(one_reference)
    return prediction_list, reference_list

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_path_prefix", type=str, help="e.g. /data/translation/")
    parser.add_argument("--evaluation_perl_script_path", type=str)
    parser.add_argument("--benchmark_name", type=str, help="e.g. iwslt14")
    parser.add_argument("--translation_direction", type=str)
    parser.add_argument("--save_path_prefix", type=str)
    parser.add_argument("--decoding_len", type=int)
    parser.add_argument("--decoding_method", type=str, help="beam, nucleus, or contrastive")
    parser.add_argument("--shot", type=int)
    parser.add_argument("--split_num", type=int, default=-1)
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    print ('Loading model...')
    # load tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.convert_tokens_to_ids(['Ċ'])[0] # 'Ċ' stands for '\n'
    # load model
    from simctg.simctgopt import SimCTGOPT
    model = SimCTGOPT(args.model_name)
    if cuda_available:
        model = model.to(device)
    model.eval()
    opt_name = args.model_name[9:]
    print ('OPT name is {}'.format(opt_name))
    print ('Model loaded.')

    save_path_prefix = args.save_path_prefix + '/{}/{}/{}-shot/{}/{}/'.format(args.benchmark_name, 
        args.translation_direction, args.shot, opt_name, args.decoding_method)

    import os
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)

    save_name = '{}-{}-{}-shot-run-{}.json'.format(opt_name, args.decoding_method, 
            args.shot, args.split_num)

    save_path = save_path_prefix + save_name
    print ('Result saving path is {}'.format(save_path))

    print ('Loading data...')
    from dataclass import Data
    full_dataset_path_prefix = args.dataset_path_prefix + '/' + args.benchmark_name + '/'
    train_path, test_path = parse_data_path(full_dataset_path_prefix, args.translation_direction, args.shot)
    data = Data(tokenizer, args.shot, test_path=test_path, train_path=train_path, split_num=args.split_num)
    print ('Data loaded.')

    print ('Performing inference...')
    data_num = len(data.reference_list)
    p = progressbar.ProgressBar(data_num)
    p.start()
    result_list = []
    with torch.no_grad():
        for index in range(data_num):
            p.update(index)
            with torch.no_grad():
                one_res_dict = inference_one_instance(args, data, index, model, tokenizer, 
                    bos_token_id, eos_token_id, cuda_available, device)
            result_list.append(one_res_dict)
    p.finish()
    print ('Inference completed!')

    print ('Saving inference result...')
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
    print ('Inference result saved.')

    print ('Performing evaluation...')
    prediction_list, reference_list = parse_prediction_result(save_path)
    from compute_bleu import compute_bleu_scores
    bleu_score = compute_bleu_scores(prediction_list, reference_list, 
        evaluation_path=args.evaluation_perl_script_path)
    bleu_score = round(bleu_score, 2)

    result_dict = {
    'bleu_score':bleu_score
    }

    print ('BLEU score is {}'.format(bleu_score))

    evaluation_save_name = save_name[:-5] + '_bleu_result.json'
    evaluation_save_path = save_path_prefix + '/' + evaluation_save_name
    with open(evaluation_save_path, 'w') as outfile:
        json.dump(result_dict, outfile, indent=4)
    print ('Evaluation completed!')

