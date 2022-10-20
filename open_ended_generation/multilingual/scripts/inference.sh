CUDA_VISIBLE_DEVICES=0 python ../inference.py\
    --data_path ../../../data/wit/wit_test_set.json\
    --language_code \
    --decoding_method \
    --prefix_len 16\
    --decoding_len 64\
    --save_path_prefix ../inference_results/