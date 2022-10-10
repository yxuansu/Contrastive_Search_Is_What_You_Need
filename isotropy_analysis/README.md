## Measuring the Isotropy of Language Models

****

<span id='all_catelogue'/>

### Catalogue:
* <a href='#data_preparation'>1. Data Preparation</a>
* <a href='#measuring'>2. Measure Isotropy</a>
* <a href='#language_code_and_model_card'>3. Language Code and Model Card</a>

****
<span id='data_preparation'/>

#### 1. Data Preparation: <a href='#all_catelogue'>[Back to Top]</a>
Before running the experiments, please make sure you have downloaded the WIT dataset as instructed [[here]](../data/README.md#1-wit-benchmark).


****
<span id='measuring'/>

#### 2. Measure Isotropy: <a href='#all_catelogue'>[Back to Top]</a>
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


|Language|Model Architecture|Language Code|Model Name|Model Size|Model Card|Isotropy|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|`English`|`GPT`|`en`|`gpt2`<br>`gpt2-medium`<br>`gpt2-large`<br>`gpt2-xl`|`117M`<br>`345M`<br>`774M`<br>`1.6B`|[[link]](https://huggingface.co/gpt2)<br>[[link]](https://huggingface.co/gpt2-medium)<br>[[link]](https://huggingface.co/gpt2-large)<br>[[link]](https://huggingface.co/gpt2-xl)|`0.10`<br>`0.25`<br>`0.70`<br>`0.72`|
|`English`|`GPT-Neo`|`en`|`EleutherAI/gpt-neo-125M`<br>`EleutherAI/gpt-neo-1.3B`<br>`EleutherAI/gpt-neo-2.7B`|`125M`<br>`1.3B`<br>`2.7B`|[[link]](https://huggingface.co/EleutherAI/gpt-neo-125M)<br>[[link]](https://huggingface.co/EleutherAI/gpt-neo-1.3B)<br>[[link]](https://huggingface.co/EleutherAI/gpt-neo-2.7B)|`0.68`<br>`0.55`<br>`0.60`|
|`English`|`OPT`|`en`|`facebook/opt-125m`<br>`facebook/opt-350m`<br>`facebook/opt-1.3b`<br>`facebook/opt-2.7b`<br>`facebook/opt-6.7b`<br>`facebook/opt-13b`<br>`facebook/opt-30b`|`125M`<br>`350M`<br>`1.3B`<br>`2.7B`<br>`6.7B`<br>`13B`<br>`30B`|[[link]](https://huggingface.co/facebook/opt-125m)<br>[[link]](https://huggingface.co/facebook/opt-350m)<br>[[link]](https://huggingface.co/facebook/opt-1.3b)<br>[[link]](https://huggingface.co/facebook/opt-2.7b)<br>[[link]](https://huggingface.co/facebook/opt-6.7b)<br>[[link]](https://huggingface.co/facebook/opt-13b)<br>[[link]](https://huggingface.co/facebook/opt-6.7b)<br>[[link]](https://huggingface.co/facebook/opt-30b)|`0.75`<br>`0.69`<br>`0.75`<br>`0.74`<br>`0.70`<br>`0.66`<br>`0.68`|
|`Spanish`|`GPT`|`es`|`datificate/gpt2-small-spanish`<br>`DeepESP/gpt2-spanish-medium`|`117M`<br>`345M`|[[link]](https://huggingface.co/datificate/gpt2-small-spanish)<br>[[link]](https://huggingface.co/DeepESP/gpt2-spanish-medium)|`0.77`<br>`0.76`|
|`French`|`GPT`|`fr`|`asi/gpt-fr-cased-small`|`117M`|[[link]](https://huggingface.co/asi/gpt-fr-cased-small)|`0.76`|
|`Portuguese`|`GPT`|`pt`|`pierreguillou/gpt2-small-portuguese`|`117M`|[[link]](https://huggingface.co/pierreguillou/gpt2-small-portuguese)|`0.77`|
|`Thai`|`GPT`|`th`|`flax-community/gpt2-base-thai`|`117M`|[[link]](https://huggingface.co/flax-community/gpt2-base-thai)|`0.74`|
|`Japanese`|`GPT`|`ja`|`colorfulscoop/gpt2-small-ja`|`117M`|[[link]](https://huggingface.co/colorfulscoop/gpt2-small-ja)|`0.72`|
|`Korean`|`GPT`|`ko`|`skt/kogpt2-base-v2`<br>`skt/ko-gpt-trinity-1.2B-v0.5`|`117M`<br>`1.6B`|[[link]](https://huggingface.co/skt/kogpt2-base-v2/tree/main)<br>[[link]](https://huggingface.co/skt/ko-gpt-trinity-1.2B-v0.5)|`0.58`<br>`0.68`|
|`Chinese`|`GPT`|`zh`|`uer/gpt2-chinese-cluecorpussmall`|`117M`|[[link]](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)|`0.66`|
|`Indonesian`|`GPT`|`id`|`cahya/gpt2-small-indonesian-522M`<br>`flax-community/gpt2-medium-indonesian`<br>`cahya/gpt2-large-indonesian-522M`|`117M`<br>`345M`<br>`774M`|[[link]](https://huggingface.co/cahya/gpt2-small-indonesian-522M)<br>[[link]](https://huggingface.co/flax-community/gpt2-medium-indonesian)<br>[[link]](https://huggingface.co/cahya/gpt2-large-indonesian-522M/tree/main)|`0.66`<br>`0.67`<br>`0.81`|
|`Bengali`|`GPT`|`bn`|`flax-community/gpt2-bengali`|`117M`|[[link]](https://huggingface.co/flax-community/gpt2-bengali)|`0.62`|
|`Hindi`|`GPT`|`hi`|`surajp/gpt2-hindi`|`117M`|[[link]](https://huggingface.co/surajp/gpt2-hindi)|`0.62`|
|`Arabic`|`GPT`|`ar`|`akhooli/gpt2-small-arabic`<br>`aubmindlab/aragpt2-medium`|`117M`<br>`345M`|[[link]](https://huggingface.co/akhooli/gpt2-small-arabic)<br>[[link]](https://huggingface.co/aubmindlab/aragpt2-medium)|`0.53`<br>`0.64`|
|`German`|`GPT`|`de`|`ml6team/gpt2-small-german-finetune-oscar`<br>`ml6team/gpt2-medium-german-finetune-oscar`|`117M`<br>`345M`|[[link]](https://huggingface.co/ml6team/gpt2-small-german-finetune-oscar)<br>[[link]](https://huggingface.co/ml6team/gpt2-medium-german-finetune-oscar)|`0.83`<br>`0.81`|
|`Dutch`|`GPT`|`nl`|`ml6team/gpt2-small-dutch-finetune-oscar`<br>`ml6team/gpt2-medium-dutch-finetune-oscar`|`117M`<br>`345M`|[[link]](https://huggingface.co/ml6team/gpt2-small-dutch-finetune-oscar)<br>[[link]](https://huggingface.co/ml6team/gpt2-medium-dutch-finetune-oscar)|`0.80`<br>`0.79`|
|`Russian`|`GPT`|`ru`|`sberbank-ai/rugpt3small_based_on_gpt2`<br>`sberbank-ai/rugpt3medium_based_on_gpt2`<br>`sberbank-ai/rugpt3large_based_on_gpt2`|`117M`<br>`345M`<br>`774M`|[[link]](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2)<br>[[link]](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2)<br>[[link]](https://huggingface.co/sberbank-ai/rugpt3large_based_on_gpt2)|`0.67`<br>`0.72`<br>`0.77`|
|`Italian`|`GPT`|`it`|`LorenzoDeMattei/GePpeTto`|`117M`|[[link]](https://huggingface.co/LorenzoDeMattei/GePpeTto)|`0.69`|

### Acknowledgements

We thank the research community for open-sourcing these wonderful language models!
