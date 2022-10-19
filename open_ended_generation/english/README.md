## Open-ended Text Generation with English LMs

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#benchmark'>1. Evaluation Benchmark</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>
* <a href='#language_code_and_model_card'>3. Language Code and Model Card</a>

****
<span id='benchmark'/>

#### 1. Evaluation Benchmark: <a href='#all_catelogue'>[Back to Top]</a>
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
where {} is in [`greedy`, `beam`, `topk`, `nucleus`, `contrastive`]. To get the results with typical sampling, please refer to the [code](https://github.com/cimeister/typical-sampling) released by the original authors.


To measure the isotropy of a specific language model, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./inference.sh
./inference.sh
```

The arguments are as follows:
* `--test_path`: The file path of the test data.
* `--max_len`: The maximum length of a test sequence.
* `--language_code`: The language code of the specific language model. See <a href='#language_code_and_model_card'>Section 3</a> for more details.
* `--model_name`: The model name of the huggingface model. See <a href='#language_code_and_model_card'>Section 3</a> for more details.
* `--save_path_prefix`: The directory used to save the inferenced result.


**[Note]** After completing the inference, the inferenced result will be saved in the directory of `save_path_prefix + r'/{}/'.format(language_code)`.

****
<span id='language_code_and_model_card'/>

#### 3. Language Code and Model Card: <a href='#all_catelogue'>[Back to Top]</a>
In the following Table, we provide the models that we use in our experiments. 




### Acknowledgements

We thank the research community for open-sourcing these wonderful language models!
