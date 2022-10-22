## Code Generation

****

### Catalogue:
* <a href='#evaluation_setup'>1. Evaluation Setup</a>
* <a href='#inference'>2. Inference with Different Decoding Methods</a>

****
<span id='evaluation_setup'/>

#### 1. Evaluation Setup:

First, we should install the evaluation environment `human-eval` for HumanEval benchmark following the official instructions [[here]](https://github.com/openai/human-eval). After installation, please run a quick sanity check as suggested in the [official instructions](https://github.com/openai/human-eval#usage).

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
* `--test_path`: The file path of the test data.
* `--decoding_len`: The number of generated tokens for each instance.
* `--run_num`: The number of evaluation runs. It should be set as [`1`, `2`, `3`], respectively, if the user would like to test stochastic nucleus sampling for multiple (e.g. 3) runs.
* `--evaluation_method`: The decoding method that used to generate the result and it should be one of [`beam`, `nucleus`, `contrastive`].
* `--model_name`: The CodeGen model that used to generate the result and it should be one of [`Salesforce/codegen-350M-mono`, `Salesforce/codegen-2B-mono`]
* `--save_path_prefix`: The directory used to save the inferenced result.

After completing the inference, the inferenced and evaluated results will be saved in the directory of `save_path_prefix + r'/{}/{}/'.format(model_name, 
        evaluation_method)`.

**[Reproducibility]** To make our experiments precisely reproducible, we have provided all our inferenced results in the folder `./inference_results/`.
