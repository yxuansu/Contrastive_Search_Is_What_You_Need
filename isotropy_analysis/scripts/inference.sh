CUDA_VISIBLE_DEVICES=0 python ../compute_language_model_isotropy.py\
    --test_path ../../data/wit/wit_test_set.json\
    --max_len 256\
    --language_code \
    --model_name \
    --save_path_prefix ../inference_results/
