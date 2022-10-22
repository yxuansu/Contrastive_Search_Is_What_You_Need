import os
import subprocess
from subprocess import call

def eval_bleu(reference_file_path, predictions_file_path, evaluation_path):
    perl_path = evaluation_path + '/multi-bleu.perl'
    command = 'perl '+ perl_path + ' ' + reference_file_path + ' ' + '<' + ' ' + predictions_file_path
    result = subprocess.run(command,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,)
    res = result.stdout.decode("utf-8") 
    return float(res.split()[2].strip(','))

def compute_bleu_scores(prediction_list, reference_list, evaluation_path):
    path_prefix = r'./tmp_bleu/'
    import os
    if os.path.exists(path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(path_prefix, exist_ok=True)

    pred_path = path_prefix + '/predictions.txt'
    with open(pred_path, 'w', encoding = 'utf8') as o:
        for text in prediction_list:
            o.writelines(text + '\n')

    ref_path = path_prefix + '/references.txt'
    with open(ref_path, 'w', encoding = 'utf8') as o:
        for text in reference_list:
            o.writelines(text + '\n')
    bleu_score = eval_bleu(ref_path, pred_path, evaluation_path)

    os.system(r'rm -r ./tmp_bleu/')
    return bleu_score
