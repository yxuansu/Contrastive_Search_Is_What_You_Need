import json
import torch
import progressbar

class Data:
    def __init__(self, tokenizer, test_path, language_code, prefix_len, decoding_len, test_num):
        self.tokenizer = tokenizer
        self.prefix_len, self.decoding_len = prefix_len, decoding_len
        self.min_len = self.prefix_len + self.decoding_len
        self.test_num = test_num
        self.prefix_token_id_list, self.prefix_text_list, self.reference_text_list = \
        self.extract_data(test_path, language_code)
        print ('Number of test samples is {}'.format(len(self.prefix_token_id_list)))

    def extract_data(self, test_path, language_code):
        with open(test_path) as f:
            text_dict = json.load(f)
            text_list = text_dict[language_code]

        prefix_token_id_list, prefix_text_list, reference_text_list = [], [], []
        data_num = len(text_list)
        for idx in range(data_num):
            if len(prefix_token_id_list) == self.test_num:
                break
            text = text_list[idx].strip('\n').strip()
            self.process_one_text(text, prefix_token_id_list, prefix_text_list, reference_text_list)
        return prefix_token_id_list, prefix_text_list, reference_text_list

    def process_one_text(self, text, prefix_token_id_list, prefix_text_list, reference_text_list):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < self.min_len:
            return

        # tokenize text
        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        # get prefix id list
        prefix_id_list = token_id_list[:self.prefix_len]
        prefix_token_id_list.append(prefix_id_list)
        # get prefix text
        prefix_text = self.tokenizer.decode(prefix_id_list)
        prefix_text_list.append(prefix_text)
        # get reference text
        reference_text = self.tokenizer.decode(token_id_list[self.prefix_len:])
        reference_text_list.append(reference_text)
        return