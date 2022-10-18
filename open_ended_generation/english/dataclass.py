import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn

class Data:
    def __init__(self, tokenizer, prefix_len, decoding_len, data_path):
        '''
            data_path: the path of test data
            prefix_len: length of the human-written prefix
            decoding_len: length of generated text continuation
        '''
        self.tokenizer = tokenizer
        self.prefix_len, self.decoding_len = prefix_len, decoding_len
        self.min_len = self.prefix_len + self.decoding_len

        self.prefix_token_id_list, self.prefix_text_list, self.reference_text_list = \
        self.process_one_file(data_path)
        print ('Evaluation number is {}'.format(len(self.prefix_token_id_list)))

    def process_one_file(self, path):
        print ('Processing {}'.format(path))
        prefix_token_id_list, prefix_text_list, reference_text_list = [], [], []

        import json
        with open(path) as f:
            data = [json.loads(line) for line in f]
        n = len(data)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            item = data[i]
            text = item['text']
            self.process_one_text(text, prefix_token_id_list, prefix_text_list, reference_text_list)
        p.finish()
        print ('{} processed!'.format(path))
        return prefix_token_id_list, prefix_text_list, reference_text_list

    def process_one_text(self, text, prefix_token_id_list, prefix_text_list, reference_text_list):
        tokens = self.tokenizer.tokenize(text)
        total_len = self.prefix_len + self.decoding_len
        if len(tokens) < total_len:
            return

        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        prefix_id_list = token_id_list[:self.prefix_len]
        prefix_token_id_list.append(prefix_id_list)
        prefix_text = self.tokenizer.decode(prefix_id_list)
        prefix_text_list.append(prefix_text)
        reference_id_list = token_id_list[self.prefix_len:total_len]
        reference_text = self.tokenizer.decode(reference_id_list)
        reference_text_list.append(reference_text)
        return 
