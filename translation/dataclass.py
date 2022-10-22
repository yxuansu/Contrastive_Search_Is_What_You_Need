import json
import torch
import progressbar

class Data:
    def __init__(self, tokenizer, shot, test_path, train_path=None, split_num=1):
        # shot: number of in-context learning training examples
        train_context = ''
        if shot != 0:
            assert train_path != None
            assert split_num > 0
            with open(train_path) as f:
                train_data = json.load(f)

            start_idx, end_idx = shot*(split_num-1), shot*split_num
            for item in train_data[start_idx:end_idx]:
                train_context += item['context'] + item['completion']
        else:
            pass
        self.train_context = train_context

        print ('Loading test data...')
        with open(test_path) as f:
            data = json.load(f)
        #data_num = len(data)
        data_num = 1000

        self.context_list, self.context_token_id_list, self.reference_list = [], [], []
        p = progressbar.ProgressBar(data_num)
        p.start()
        for idx in range(data_num):
            p.update(idx)
            item = data[idx]
            context = self.train_context + item['context']
            self.context_list.append(item['context'])
            context_token_list = tokenizer.tokenize(context)
            context_token_id_list = tokenizer.convert_tokens_to_ids(context_token_list)
            self.context_token_id_list.append(context_token_id_list)
            reference = item['completion'].strip()
            self.reference_list.append(reference)
        p.finish()
        print ('Number of test data is {}'.format(len(self.reference_list)))
