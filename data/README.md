
## Datasets

****

### Catalogue:
* <a href='#wit'>1. WIT Benchmark</a>
* <a href='#xsum'>2. XSum Benchmark</a>
* <a href='#translation'>3. IWSLT14 De-En Benchmark</a>
* <a href='#webtext'>4. WebText Benchmark</a>


****
<span id='wit'/>

#### 1. WIT Benchmark:

Unzip the file (wit.zip) under the current directory.


> **** The resulting directory looks like

    .
    ├── ./wit/                    
        └── wit_test_set.json # The test of WIT dataset


****
<span id='xsum'/>

#### 2. XSum Benchmark:
We provide the in-context learning examples as well as the test set of XSum benchmark that are used in our experiments.

> **** The directory looks like

    .
    ├── ./xsum/                    
        ├── /one-shot/ # Contains the one-shot in-context examples.
            ├── xsum_train_one-shot-1.json # The first random one-shot in-context learning example.
            ├── xsum_train_one-shot-2.json # The second random one-shot in-context learning example.
            └── xsum_train_one-shot-3.json # The third random one-shot in-context learning example.
        ├── /two-shot/ # Contains the two-shot in-context examples
            ├── xsum_train_two-shot-1.json # The first random two-shot in-context learning example.
            ├── xsum_train_two-shot-2.json # The second random two-shot in-context learning example.
            └── xsum_train_two-shot-3.json # The third random two-shot in-context learning example.
        └── xsum_test.json # The test set of XSum benchmark.
        
        
****
<span id='translation'/>

#### 3. IWSLT14 De-En Benchmark:
We provide the IWSLT14 De-En dataset that is used in our experiments. 

> **** After unzipping, the directory looks like

    .
    ├── ./translation/iwslt14/de-to-en/                    
        ├── train.json # The training set of IWSLT14 De-En dataset.
        ├── validation.json # The validation set of IWSLT14 De-En dataset.
        └── test.json # The test set of IWSLT14 De-En dataset.
 
****
<span id='webtext'/>

#### 4. WebText Benchmark:
 
We provide the held-out set of WebText.

> **** The directory looks like

    .
    ├── ./webtext/                    
        └── webtext.test.jsonl # The held-out set of the WebText benchmark.
