## Document Summarization

****

### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#evaluation_setup'>2. Evaluation Setup</a>
* <a href='#inference'>3. Inference with Different Decoding Methods</a>

****
<span id='data_preparation'/>

#### 1. Data Preparation:
Before running the experiments, please make sure you have downloaded the XSum dataset as provided [[here]](../data/xsum/). 


> ****  The structure of the provided dataset looks like:

    .
    ├──
        ├── ../data/xsum/xsum_test.json  # The test set of XSum benchmark.
        ├── ../data/xsum/one-shot/  # This directory contains three random sets of one-shot training data of XSum.
        └── ../data/xsum/two-shot/  # This directory contains three random sets of two-shot training data of XSum.

****
<span id='evaluation_setup'/>

#### 2. Evaluation Setup:

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
* `--evaluation_method`: The decoding method that used to generate the result and it should be one of [`greedy`, `beam`, `nucleus`, `contrastive`].
* `--model_name`: The CodeGen model that used to generate the result and it should be one of [`Salesforce/codegen-350M-mono`, `Salesforce/codegen-2B-mono`]
* `--save_path_prefix`: The directory used to save the inferenced result.

**[Note]** After completing the inference, the inferenced and evaluated results will be saved in the directory of `save_path_prefix + r'/{}/{}/'.format(model_name, 
        evaluation_method)`.
