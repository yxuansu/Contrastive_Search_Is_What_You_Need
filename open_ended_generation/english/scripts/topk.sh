CUDA_VISIBLE_DEVICES=1 python ../inference.py\
    --model_name gpt2-large\
    --data_path ../../../data/webtext/webtext.test.jsonl\
    --decoding_method topk\
    --number_of_instance_to_generate_per_method 3\
    --prefix_len 40\
    --decoding_len 200\
    --save_path_prefix ../inference_results/