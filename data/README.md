
## Datasets

****

### Catalogue:
* <a href='#wit'>1. WIT Benchmark</a>
* <a href='#xsum'>2. XSum Benchmark</a>


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
