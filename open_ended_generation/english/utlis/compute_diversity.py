import json
import torch
import argparse
import numpy as np

def load_result(in_f):
    with open(in_f) as f:
        result_list = json.load(f)

    # load reference list
    reference_list = []
    for item in result_list:
        one_reference_text = item['reference_text']
        reference_list.append(one_reference_text)

    # load all predictions
    number_of_predictions_per_instance = len(result_list[0]['generated_result'])
    print ('Number of predictions per instance is {}'.format(number_of_predictions_per_instance))
    all_prediction_list = []
    for idx in range(number_of_predictions_per_instance):
        one_prediction_list = []
        for item in result_list:
            one_prediction = item['generated_result'][str(idx)]
            one_prediction_list.append(one_prediction)
        assert len(one_prediction_list) == len(reference_list)
        all_prediction_list.append(one_prediction_list)
    return reference_list, all_prediction_list

def measure_diversity(in_f):
    reference_list, all_prediction_list = load_result(in_f)
    from simctg.evaluation import measure_repetition_and_diversity
    _, _, _, reference_diversity = measure_repetition_and_diversity(reference_list)
    reference_diversity = round(reference_diversity*100, 2)

    prediction_diversity_list = []
    for idx in range(len(all_prediction_list)):
        _, _, _, one_prediction_diversity = measure_repetition_and_diversity(all_prediction_list[idx])
        one_prediction_diversity = round(one_prediction_diversity*100, 2)
        prediction_diversity_list.append(one_prediction_diversity)

    pred_div_mean = np.mean(prediction_diversity_list)
    pred_div_std = np.std(prediction_diversity_list)

    result_dict = {
        'reference_div': str(reference_diversity),
        'prediction_diversity_list': [str(num) for num in prediction_diversity_list],
        'prediction_div_mean': str(pred_div_mean),
        'prediction_div_std': str(pred_div_std),
    }
    return result_dict
