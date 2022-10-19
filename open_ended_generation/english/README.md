## Open-ended Text Generation with English LMs

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#benchmark'>1. Benchmark</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>
* <a href='#evaluation'>3. Evaluation</a>
    * <a href='#diversity_mauve_gen_length'>3.1. Diversity, MAUVE, Generation Length</a>


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


**[Note]** After completing the inference, the inferenced result will be saved in the directory of `save_path_prefix + r'/{}/{}/{}_result.json'.format(model_name, decoding_method, decoding_method)`.

**[Reproducibility]** To make our experiments precisely reproducible, we have all the inferenced results in the folder `./inference_results/`.

****
<span id='evaluation'/>

#### 3. Evaluation: <a href='#all_catelogue'>[Back to Top]</a>

<span id='diversity_mauve_gen_length'/>

##### 3.1. Diversity, MAUVE, Generation Length: 




### Acknowledgements

We thank the research community for open-sourcing these wonderful language models!
