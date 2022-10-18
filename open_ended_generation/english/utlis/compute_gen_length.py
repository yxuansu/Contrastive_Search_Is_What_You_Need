import json
import numpy as np
def load_result(in_f):
    with open(in_f) as f:
        result_list = json.load(f)

    # load all predictions
    number_of_predictions_per_instance = len(result_list[0]['generated_result'])
    print ('Number of predictions per instance is {}'.format(number_of_predictions_per_instance))
    all_prediction_list = []
    for idx in range(number_of_predictions_per_instance):
        one_prediction_list = []
        for item in result_list:
            one_prediction = item['generated_result'][str(idx)]
            one_prediction_list.append(one_prediction)
        all_prediction_list.append(one_prediction_list)
    return all_prediction_list
    
def compute_one_gen_len(text_list):
    all_len = 0.
    for text in text_list:
        all_len += len(text.strip().split())
    return all_len / len(text_list)
    
def measure_gen_length(in_f):
    all_prediction_list = load_result(in_f)
    len_list = []
    for one_prediction_list in all_prediction_list:
        len_list.append(compute_one_gen_len(one_prediction_list))
    result_dict = {
        'gen_len_list': [str(num) for num in len_list],
        'gen_len_mean': str(round(np.mean(len_list),2)),
        'gen_len_std': str(round(np.std(len_list),2))
    }
    return result_dict