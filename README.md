# Pseudo-Perplexity Evaluation Function

## Overview

Pseudo-perplexity can be effectively applied to RoBERTa models, serving as a useful metric for evaluating the model's performance on various tasks. Here are some key points about applying pseudo-perplexity to RoBERTa:

1. Evaluation metric: Pseudo-perplexity is used to assess how well RoBERTa models predict a sample or represent a corpus of sentences[1][3].

2. Calculation method: For RoBERTa, pseudo-perplexity is computed using masked language modeling. This involves masking tokens one by one and calculating the conditional probabilities of each token given the surrounding context[3].

3. Performance indicator: Lower pseudo-perplexity scores indicate better model performance, suggesting that the RoBERTa model predicts the sequence well[1][3].

4. Comparison with other models: RoBERTa models using pseudo-perplexity have shown competitive results compared to other language models. For instance, in some tasks, RoBERTa with pseudo-perplexity outperformed autoregressive language models like GPT-2[2].

5. Application in various tasks: Pseudo-perplexity has been applied to RoBERTa models for tasks such as evaluating grammatical correctness, assessing language model quality, and comparing performance across different languages and tokenizers[1][4].

6. Sliding window strategy: When calculating pseudo-perplexity for RoBERTa models, a sliding window approach can be used to provide more context for each prediction, potentially improving the reported perplexity scores[5].

By utilizing pseudo-perplexity, researchers and practitioners can gain valuable insights into the performance and capabilities of RoBERTa models across various natural language processing tasks.

Citations:

[1] https://aclanthology.org/2024.emnlp-main.638.pdf

[2] https://aclanthology.org/2021.naacl-main.158.pdf

[3] https://assets.amazon.science/cc/54/980cb7d74f93849b49b9d5c42466/masked-language-model-scoring.pdf

[4] https://www.scribendi.ai/comparing-bert-and-gpt-2-as-language-models-to-score-the-grammatical-correctness-of-a-sentence/

[5] https://huggingface.co/docs/transformers/perplexity?highlight=perplexity

[6] https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode

[7] https://github.com/asahi417/lmppl

[8] https://www.researchgate.net/figure/Pseudo-perplexity-of-the-selected-tasks-compared-to-pre-training-data-from-Wikipedia-red_fig7_384211206

## Run the code

As part of DT4H WP3, we have implemented a pseudo-perplexity evaluation function for RoBERTa models. The code is available in the `ppl.py` file in this repository. The approach followed in this work consists on calculating the pseudo-perplexity of a RoBERTa model using the Paraclite dataset.

- **Step 1**: the segments corresponding to the same document are concatenated and converted into a single sequence.
- **Step 2**: the sequence is tokenized using the RoBERTa tokenizer and chunked using a sliding window approach.
- **Step 3**: the pseudo-perplexity is calculated for each chunk.

 To run the code, follow these steps:

1. Download paraclite dataset from Huggingface datasets

```bash
huggingface-cli download DT4H/paraclite --repo-type dataset --local-dir data
```

2. Install the required dependencies

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the evaluation function

```bash
python ppl.py \
    --model "/gpfs/projects/bsc14/abecerr1/hub/models--PlanTL-GOB-ES--roberta-base-biomedical-clinical-es/snapshots/c6bfaa3cc4453dc6d947d279e3905c7083663af1/" \
    --csv_path "data/data/paraclite.csv" \
    --language "es"
```

A new folder named `output` will be created with the results of the evaluation. Negative log-likelihood and perplexity scores will be saved in a CSV file.

For any questions or issues, please contact Alberto Becerra (abecerr1bsc@gmail.com).

# TODO

* Add [LMPPL](https://github.com/asahi417/lmppl) as backend
