import os
import subprocess
from subprocess import call

def evaluate_pass_score(problem_file_path, prediction_file_path):
    command = 'evaluate_functional_correctness ' + prediction_file_path + ' --problem_file=' + problem_file_path
    result = subprocess.run(command,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,)
    res = result.stdout.decode("utf-8") 
    score = float(res.split(r"{'pass@1': ")[1].split('}')[0])
    return score

if __name__ == '__main__':
    #problem_file_path = r'./evaluation_debug/human-eval/data/example_problem.jsonl'
    #prediction_file_path = r'./evaluation_debug/human-eval/data/example_samples-2.jsonl'
    problem_file_path = r'./HumanEval.jsonl'
    prediction_file_path = r'./inference_results/codegen-350M-mono/greedy/codegen-350M-mono-greedy-run-1.jsonl'
    print (evaluate_pass_score(problem_file_path, prediction_file_path))
