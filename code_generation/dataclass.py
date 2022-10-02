import json
import torch
import progressbar

class Data:
    def __init__(self, test_path, tokenizer):
        with open(test_path, 'r') as json_file:
            json_list = list(json_file)

        question_list = []
        for json_str in json_list:
            question_list.append(json.loads(json_str))

        question_token_id_list = []
        self.data_list = []
        for question in question_list:
            task_id = question['task_id']
            prompt = question['prompt']
            prompt_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
            one_data = {
                'task_id': task_id,
                'prompt': prompt,
                'prompt_ids': prompt_ids
            }
            self.data_list.append(one_data)

