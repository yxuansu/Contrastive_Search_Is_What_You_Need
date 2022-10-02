## Code Generation

****

### Catalogue:
* <a href='#evaluation_setup'>1. Evaluation Setup</a>
* <a href='#train_simctg'>2. Train SimCTG</a>
* <a href='#inference'>3. Inference with SimCTG</a>
* <a href='#generate_results'>4. Generate Result with Different Decoding Methods</a>
    * <a href='#contrastive_search'>4.1. Contrastive Search</a>
    * <a href='#diverse_contrastive_search'>4.2. Diverse Contrastive Search</a>
    * <a href='#nucleus_sampling'>4.3. Nucleus Sampling</a>
    * <a href='#greedy_search'>4.4. Greedy Search</a>
    * <a href='#beam_search'>4.5. Beam Search</a>
* <a href='#evaluation'>5. Evaluate the Generated Text</a>
* <a href='#visualize_token_similarity_matrix'>6. Visualize the Token Similarity Matrix</a>

****
<span id='evaluation_setup'/>

#### 1. Evaluation Setup:

First, we should install the evaluation environment `human-eval` for HumanEval benchmark following the official instructions [[here]](https://github.com/openai/human-eval). After installation, please run a quick sanity check as suggested in the [official instructions](https://github.com/openai/human-eval#usage).
