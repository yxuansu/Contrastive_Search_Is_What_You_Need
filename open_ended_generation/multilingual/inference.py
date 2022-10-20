# coding=utf-8
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

def inference_one_instance(args, data, index, eos_token_id, model, cuda_available, device, 
    contrastive_dict):
    decoding_method = args.decoding_method
    assert decoding_method in ['nucleus', 'contrastive']

    input_ids = data.prefix_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    _, prefix_len = input_ids.size()
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_len = args.decoding_len
    all_output_text_list = []
    with torch.no_grad():
        if decoding_method == 'contrastive':
            number_of_instance_to_generate_per_method = 1
            k, alpha = contrastive_dict['k'], contrastive_dict['alpha']
            output = model.fast_contrastive_search(input_ids=input_ids, beam_width=k, alpha=alpha, 
                        decoding_len=decoding_len, end_of_sequence_token_id = eos_token_id, early_stop = True)
            output_text = model.tokenizer.decode(output[prefix_len:])
            all_output_text_list = [output_text]
        else:
            number_of_instance_to_generate_per_method = 1
            output = model.nucleus_sampling(input_ids=input_ids, nucleus_p=0.95, decoding_len=decoding_len, 
                end_of_sequence_token_id = eos_token_id, early_stop = True)
            output_text = model.tokenizer.decode(output[prefix_len:])
            all_output_text_list.append(output_text)

    res_dict = {}
    res_dict['prefix_text'] = data.prefix_text_list[index]
    res_dict['reference_text'] = data.reference_text_list[index]

    generated_dict = {}
    for one_idx in range(number_of_instance_to_generate_per_method):
        generated_dict[one_idx] = all_output_text_list[one_idx]
    res_dict['generated_result'] = generated_dict
    return res_dict

def inference_one_file(args, data, save_path, model, eos_token_id, cuda_available, device):
    print ('----------------------------------------------------------------')
    print ('Start inference...')
    test_num = len(data.prefix_token_id_list)

    # for debugging purpose only
    #test_num = 10
    
    result_list = []
    p = progressbar.ProgressBar(test_num)
    p.start()
    with torch.no_grad():
        for index in range(test_num):
            p.update(index)
            one_res_dict = inference_one_instance(args, data, index, model, eos_token_id, cuda_available, device)
            result_list.append(one_res_dict)
        p.finish()
    import json
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
    print ('Inference completed.')

def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--language_code", type=str)
    # decoding configuration
    parser.add_argument("--decoding_method", type=str)
    parser.add_argument("--prefix_len", type=int)
    parser.add_argument("--decoding_len", type=int)
    # save configuration
    parser.add_argument("--save_path_prefix", type=str)
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    save_path_prefix = args.save_path_prefix + '{}/{}/'.format(args.language_code, args.decoding_method)
    import os
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = '{}_result.json'.format(args.decoding_method)
    save_path = save_path_prefix + save_name
    print ('Result saving path is {}'.format(save_path))

    from utlis import Contrastive_Dict
    contrastive_dict = Contrastive_Dict[args.language_code]
    print ('Loading model...')
    from simctg.simctggpt import SimCTGGPT
    print ('Language code is {}, model name is {}'.format(args.language_code, contrastive_dict['model_name']))
    model = SimCTGGPT(contrastive_dict['model_name'])
    tokenizer = model.tokenizer
    if cuda_available:
        model = model.cuda(device)
    model.eval()
    print ('Model loaded') 

    language_code = args.language_code
    if language_code == 'zh':
        eos_token = '[SEP]'
    elif language_code == 'ko':
        eos_token = '</d>'
    else:
        eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    print ('Loading inference data...')
    from data_extractor import Data
    data = Data(tokenizer, args.data_path, args.language_code, args.prefix_len, args.decoding_len, 100)
    print ('Inference data loaded.')

    print ('----------------------------------------------------------------')
    print ('Start inference...')
    test_num = len(data.prefix_token_id_list)
    result_list = []
    p = progressbar.ProgressBar(test_num)
    p.start()
    with torch.no_grad():
        for index in range(test_num):
            p.update(index)
            one_res_dict = inference_one_instance(args=args, data=data, index=index, eos_token_id=eos_token_id, 
                model=model, cuda_available=cuda_available, device=device, contrastive_dict=contrastive_dict)
            result_list.append(one_res_dict)
        p.finish()
    print ('Inference completed.')
    print ('----------------------------------------------------------------')

    import json
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)
    