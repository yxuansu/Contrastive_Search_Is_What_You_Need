## Measuring the Isotropy of Language Models

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
Before running the experiments, please make sure you have downloaded the WIT dataset as instructed [[here]](../data/README.md#1-wit).
