## Machine Translation

****

### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>

****
<span id='data_preparation'/>

#### 1. Data Preparation:

Before running experiments, please make sure you have prepared the dataset following our instructions [[here]](https://github.com/yxuansu/Contrastive_Search_Is_All_You_Need/tree/main/data#3-iwslt14-de-en-benchmark).

****
<span id='inference'/>

#### 2. Inference with Different Decoding Methods:
To perform inference with different decoding methods, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./inference.sh
./inference.sh
```

The arguments are as follows:
* `--dataset_path_prefix`: The directory path that saves the dataset.
* `--evaluation_perl_script_path`: The directory path that stores the necessary evaluation script `multi-bleu.perl`.
* `--benchmark_name`: The benchmark name.
* `--translation_direction`: The direction of translation.
* `--save_path_prefix`: The directory path that saves the inferenced and evaluation results.
* `--decoding_len`: The number of generated tokens for each instance.
* `--decoding_method`: The decoding method that used to generate the result and it should be one of [`beam`, `nucleus`, `contrastive`].
* `--model_name`: The model name of the OPT model. In our experiments, it is one of [`facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`],
* `--shot`: The number of in-context examples provided to the OPT model. In our experiments, it is one of [`1`, `8`].
* `--split_num`: The split of random selection of in-context examples. In our experiments, it is one of [`1`, `2`, `3`].

After completing the inference, the inferenced and evaluated results will be saved in the directory of `save_path_prefix + '/{}/{}/{}-shot/{}/{}/'.format(benchmark_name, translation_direction, shot, model_name, decoding_method)`.

**[Reproducibility]** To make our experiments precisely reproducible, we have provided all our inferenced results in the folder `./inference_results/`.
