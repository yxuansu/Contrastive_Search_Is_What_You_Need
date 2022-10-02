CUDA_VISIBLE_DEVICES=0 python ../inference.py\
    --test_path ../HumanEval.jsonl\
    --decoding_len 128\
    --run_num 1\
    --evaluation_method \
    --model_name \
    --save_path_prefix ../inference_results/