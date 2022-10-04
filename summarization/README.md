## Document Summarization

****

### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#evaluation_setup'>2. Evaluation Setup</a>
    * <a href='#rogue_source'>2.1. Install Pyrouge from Source</a>
    * <a href='#rogue_official'>2.2. Install Official ROUGE Script</a>
    * <a href='#rogue_point'>2.3. Point Pyrouge to Official ROGUE Script</a>
    * <a href='#rogue_parser'>2.4. Install libxml Parser [Optional]</a>
    * <a href='#rogue_DB'>2.5. Regenerate the Exceptions DB</a>
    * <a href='#rogue_locale'>2.6. Fix Locale Setting</a>
    * <a href='#rogue_test'>2.7. Run Test</a>
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
Next, we need to install the evaluation environment (i.e. pyrouge) for compute ROGUE scores. Please follow the commands below as described [[here]](https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu). In the following, we provide a detailed tutorial on how to setup the evaluation environment.

**[Note]** The provided tutorial is only tested on Ubuntu systems.

<span id='rogue_source'/>

##### 2.1. Install Pyrouge from Source:
```yaml
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
```

<span id='rogue_official'/>

##### 2.2. Install Official ROUGE Script:
```yaml
git clone https://github.com/andersjo/pyrouge.git rouge
```

<span id='rogue_point'/>

##### 2.3. Point Pyrouge to Official ROGUE Script:
```yaml
pwd
```
It returns the `current_path`.

```yaml
pyrouge_set_rouge_path current_path/rouge/tools/ROUGE-1.5.5/
```

<span id='rogue_parser'/>

##### 2.4. Install libxml Parser [Optional]:
```yaml
sudo apt-get install libxml-parser-perl
```

<span id='rogue_DB'/>

##### 2.5. Regenerate the Exceptions DB:
```yaml
cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

<span id='rogue_locale'/>

##### 2.6. Fix Locale Setting:
If you meet the error below:
```yaml
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
	LANGUAGE = "en_GB:en",
	LC_ALL = (unset),
	LC_CTYPE = "UTF-8",
	LANG = "en_GB.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to a fallback locale ("en_GB.UTF-8").
```

You can then fix it as described [[here]](https://stackoverflow.com/questions/2499794/how-to-fix-a-locale-setting-warning-from-perl).
```yaml
# Setting for the new UTF-8 terminal support in Lion
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

<span id='rogue_test'/>

##### 2.7. Run Test:
```yaml
python -m pyrouge.test
```

You should see results like below:
```yaml
Ran 11 tests in 5.444s

OK
```




First, we should install the evaluation environment `human-eval` for HumanEval benchmark following the official instructions [[here]](https://github.com/openai/human-eval). After installation, please run a quick sanity check as suggested in the [official instructions](https://github.com/openai/human-eval#usage).

****
<span id='inference'/>

#### 3. Inference with Different Decoding Methods:
To perform inference with different decoding methods, please run the following commands:
```yaml
cd ./scripts/X-shot/
chmod +x ./Y.sh
./Y.sh
```
where X is in [`zero`, `one`, `two`] and Y is in [`beam`, `nucleus`, `contrastive`].

The arguments are as follows:
* `--dataset_path_prefix`: The location that stores the data of XSum benchmark.
* `--decoding_len`: The number of generated tokens for each instance.
* `--decoding_method`: The decoding method that used to generate the result and it should be one of [`greedy`, `beam`, `nucleus`, `contrastive`].
* `--model_name`: The OPT model that used to generate the result and it should be one of [`facebook/opt-125m`, `facebook/opt-350m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`].
* `--evaluation_mode`: The evaluation mode of the inference. It should be in [`zero-shot`, `one-shot`, `two-shot`].
* `--split_num`: The random split of the training set. For zero-shot evaluation, it should be set as 1. For one/two-shot evaluations, it should be in [`1`, `2`, `3`].
* `--save_path_prefix`: The directory used to save the inferenced result.

**[Note]** After completing the inference, the inferenced and evaluated results will be saved in the directory of `save_path_prefix + '/{}/{}/{}/'.format(evaluation_mode, opt_model_name, decoding_method)
