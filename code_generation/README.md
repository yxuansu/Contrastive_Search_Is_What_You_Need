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
* `--ckpt_path`: The path of trained checkpoint. You can either use our released checkpoint (`cambridgeltl/simctg_wikitext103`) or your own trained model that can be found in the `--save_path_prefix` directory as defined in train.sh.
* `--dev_path`: The file path of validation set.
* `--test_path`: The file path of test set.
* `--prefix_len`: The length of prefix.
* `--decoding_len`: The length of generated text continuation.
* `--k`: The k in contrastive search.
* `--alpha`: The \alpha in contrastive search.
* `--save_path`: Where to save the generated result.
