## Open-ended Text Generation with English LMs

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#benchmark'>1. Benchmark</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>
* <a href='#evaluation'>3. Evaluation</a>
  * <a href='#diversity_mauve_gen_length'>3.1. Diversity, MAUVE, Generation Length</a>
  * <a href='#coherence'>3.2. Coherence</a>


****
<span id='benchmark'/>

#### 1. Benchmark: <a href='#all_catelogue'>[Back to Top]</a>
We provide the held-out set of WebText benchmark at [here](../../data/webtext/).


****
<span id='inference'/>

#### 2. Inference with Different Decoding Methods: <a href='#all_catelogue'>[Back to Top]</a>

To generate text with different decoding methods, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./{}.sh
./{}.sh
```
where {} is in [`greedy`, `beam`, `topk`, `nucleus`, `contrastive`]. To get the results with typical sampling, please refer to the [original code](https://github.com/cimeister/typical-sampling) released by the authors.

The arguments are as follows:
* `--model_name`: The LMs used to generate text. In our experiments, we use `gpt2-large`.
* `--data_path`: The file path of the evaluated benchmark.
* `--decoding_method`: The decoding method used to generate text and it should be within [`greedy`, `beam`, `topk`, `nucleus`, `contrastive`].
* `--prefix_len`: The length of the prefix text.
* `--decoding_len`: The maximum length of the generated text. (The generation of text also ends upon producing the special end of document token.)
* `--save_path_prefix`: The directory used to save the inferenced result.


After completing the inference, the inferenced result will be saved in the directory of `save_path_prefix + r'/{}/{}/{}_result.json'.format(model_name, decoding_method, decoding_method)`.

**[Reproducibility]** To make our experiments precisely reproducible, we have all the inferenced results in the folder `./inference_results/`.

****
<span id='evaluation'/>

#### 3. Evaluation: <a href='#all_catelogue'>[Back to Top]</a>

<span id='diversity_mauve_gen_length'/>

##### 3.1. Diversity, MAUVE, Generation Length: 

**[Package Installation]** Please first install the required package following the [[instructions]](https://github.com/krishnap25/mauve#installation) provided by the authors.

To compute the diversity, MAUVE, and gen-ppl results, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./measure_diversity_mauve_gen_length.sh
./measure_diversity_mauve_gen_length.sh
```

The argument is as follows:
* `--test_path`: The path that stores the inferenced results from <a href='#inference'>Section 2</a>.

After the evaluation is completed, the results will be saved in the same directory as the `--test_path` (e.g. `./inference_results/gpt2-large/greedy/greedy_result_diversity_mauve_gen_length_result.json`).


<span id='coherence'/>

##### 3.2. Coherence: 

To measure the coherence score, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./measure_coherence.sh
./measure_coherence.sh
```

The argument is as follows:
* `--opt_model_name`: The name of the OPT model. In our experiments, we use one of [`facebook/opt-125m`, `facebook/opt-2.7b`, `facebook/opt-13b`].
* `--test_path`: The path that stores the inferenced results from <a href='#inference'>Section 2</a>.

After the evaluation is completed, the results will be saved in the same directory as the `--test_path` (e.g. `./inference_results/gpt2-large/greedy/greedy_result_{}_coherence_result.json`, where {} is within [`opt-125m`, `opt-2.7b`, `opt-13b`].).






