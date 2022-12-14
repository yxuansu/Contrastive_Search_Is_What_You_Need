## Open-ended Text Generation with Multilingual LMs

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#benchmark'>1. Benchmark</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>

****
<span id='benchmark'/>

#### 1. Benchmark: <a href='#all_catelogue'>[Back to Top]</a>
We provide the test set of WIT benchmark at [here](../../data/). Please follow our [instructions](https://github.com/yxuansu/Contrastive_Search_Is_All_You_Need/tree/main/data#1-wit-benchmark) to prepare the data.


****
<span id='inference'/>

#### 2. Inference with Different Decoding Methods: <a href='#all_catelogue'>[Back to Top]</a>

To generate text with different decoding methods, please run the following commands:
```yaml
cd ./scripts/
chmod +x ./inference.sh
./inference.sh
```

The arguments are as follows:
* `--data_path`: The file path of the evaluated benchmark.
* `--language_code`: The code of the evaluated language. You can refer to this [table](https://github.com/yxuansu/Contrastive_Search_Is_All_You_Need/tree/main/isotropy_analysis#3-language-code-and-model-card-back-to-top) to see codes for different languages.
* `--decoding_method`: The decoding method used to generate text and it should be within [`nucleus`, `contrastive`].
* `--prefix_len`: The length of the prefix text.
* `--decoding_len`: The maximum length of the generated text. (The generation of text also ends upon producing the special end of document token.)
* `--save_path_prefix`: The directory used to save the inferenced result.


After completing the inference, the inferenced result will be saved in the directory of `save_path_prefix + r'/{}/{}/{}_result.json'.format(language_code, decoding_method, decoding_method)`.

**[Note]** For the interest of users', we release the results generated by contrastive search in the folder `./inference_results/`. 







