# LLM Verlan

`llm_verlan` is a small research repository for studying how French language models handle verlan and other informal French variants.

The repository currently contains two main experiment tracks:

1. A focused probing pipeline for comparing verlan words with their standard French equivalents inside a Hugging Face encoder such as `camembert-base`.
2. A sentence-level informal French benchmark under `lang_informal/` for comparing standard and informal variants with masked or causal language models.

This is a research codebase, not a packaged library. The emphasis is on transparent experiments, small curated datasets, and reproducible outputs.

## Research Focus

The central question is whether French models process verlan differently from standard forms at multiple levels:

- tokenization: does the model split verlan forms into more subwords?
- attention: do target tokens show stronger intra-word fragmentation or different context integration?
- representation: do verlan and standard variants converge to similar hidden-state representations across layers?
- sentence understanding: how much do informal substitutions change model scores and sentence embeddings?

## Repository Overview

### Main probing pipeline

The main pipeline lives in `src/` and `scripts/`.

It:

- loads a tokenizer and encoder model from Hugging Face
- reads curated verlan/standard word pairs and sentence templates
- finds the token span corresponding to the target word in each sentence
- runs a forward pass with hidden states and attentions enabled
- computes tokenization, attention, and representation-based metrics
- writes CSV outputs for downstream analysis and plotting

By default, the pipeline uses `camembert-base`.

### Informal sentence benchmark

The `lang_informal/` directory contains a broader sentence-pair setup for standard versus informal French.

It includes:

- synthetic balanced datasets covering verlan, SMS-style spelling, spoken reductions, and their combinations
- scripts to build datasets
- scripts to analyze sentence pairs with masked LMs or causal LMs
- CSV outputs with token counts, pseudo-log-likelihood or perplexity, and sentence representation similarity

### Notebooks and exploratory outputs

The repository also includes notebooks for exploratory work on:

- verlan data preparation
- fastText training
- word embeddings
- OpenAI embeddings
- attention visualization

These notebooks complement the scripted pipeline but are not required for the core experiments.

## Repository Structure

```text
llm_verlan/
├── README.md
├── requirements.txt
├── artifacts/
│   ├── cache/
│   ├── figures/
│   └── models/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── verlan_base_forms.csv
│       ├── verlan_database.csv
│       ├── verlan_experiment_core.csv
│       ├── verlan_experiment_selected.csv
│       ├── verlan_freq.csv
│       ├── verlan_probe_pairs.csv
│       ├── verlan_probe_templates.csv
│       └── wiktionary_verlan_titles.csv
├── lang_informal/
│   ├── data/
│   ├── notebook/
│   └── scripts/
├── notebooks/
├── outputs/
├── results/
├── scripts/
└── src/
```

## Installation

Create a virtual environment and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` includes:

- `pandas`
- `matplotlib`
- `sentencepiece`
- `torch`
- `transformers`

When you run the model-based scripts for the first time, the selected checkpoint will be downloaded from Hugging Face.

## Quick Start

Run all commands from the repository root:

```bash
cd llm_verlan
```

### 1. Run the verlan probe experiment

```bash
python scripts/run_verlan_probe.py
```

This script uses:

- `data/processed/verlan_probe_pairs.csv`
- `data/processed/verlan_probe_templates.csv`

and writes results to:

- `outputs/verlan_probe/experiment_results.csv`
- `outputs/verlan_probe/tokenization_results.csv`
- `outputs/verlan_probe/failures.csv`
- `outputs/verlan_probe/run_metadata.json`

Key metrics include:

- number of sentence tokens
- number of target subwords
- layer-wise word-level cosine similarity
- layer-wise sentence-level cosine similarity
- outgoing attention from the target span split into `self`, `intra-word`, and `context`
- incoming attention from context into the target span

Useful options:

```bash
python scripts/run_verlan_probe.py --model-name camembert-base --limit-pairs 5
python scripts/run_verlan_probe.py --device cpu
```

### 2. Plot aggregated probe results

```bash
python scripts/plot_verlan_probe.py
```

This reads the CSV outputs above and writes summary figures to `outputs/verlan_probe/figures/`.

The plotting script also creates a subword fragmentation summary table comparing average subword counts for verlan and standard forms.

### 3. Build the balanced informal French dataset

```bash
python lang_informal/scripts/build_balanced_dataset.py \
  --output-csv lang_informal/data/paired_informal_french_dataset.csv
```

The generated dataset contains 350 rows: 50 examples for each of seven groups:

- `verlan`
- `sms`
- `spoken`
- `verlan_sms`
- `verlan_spoken`
- `sms_spoken`
- `verlan_sms_spoken`

### 4. Analyze standard vs informal sentence pairs

```bash
python lang_informal/scripts/camembert_analysis.py \
  --input-csv lang_informal/data/paired_informal_french_dataset.csv \
  --output-csv lang_informal/data/paired_informal_french_dataset_camembert_analysis.csv \
  --model-name camembert-base
```

This script:

- tokenizes each sentence
- computes masked-LM pseudo-log-likelihood for encoder checkpoints
- computes NLL and perplexity for causal LMs when applicable
- extracts mean-pooled sentence representations
- computes cosine similarity between standard and informal sentence embeddings

Built-in model aliases include:

- `camembert`
- `camembert-base`
- `camembert_v2`
- `clair-7b-0.1`

## Data Included in the Repository

### Curated probe data

The small probe set currently includes:

- `data/processed/verlan_probe_pairs.csv`: 5 verlan/standard pairs
- `data/processed/verlan_probe_templates.csv`: 12 sentence templates grouped by part-of-speech-like usage

These files are used directly by the scripted probing pipeline.

### Lexical resources

The `data/processed/` directory also includes supporting lexical tables such as:

- `verlan_database.csv`
- `verlan_base_forms.csv`
- `wiktionary_verlan_titles.csv`
- `verlan_freq.csv`

These are useful for extending the probe set or building future experiments.

### Informal sentence datasets

Under `lang_informal/data/`, the repository contains paired datasets for standard and informal French, including a second version with richer metadata and precomputed analysis outputs.

## Example Outputs Already Present

The repository already includes example outputs from previous runs, including:

- `outputs/verlan_probe/experiment_results.csv`
- `outputs/verlan_probe/tokenization_results.csv`
- `outputs/verlan_probe/run_metadata.json`
- `outputs/openai_pairwise_results.csv`
- `outputs/openai_retrieval_results.csv`
- `results/verlan_frequency_full.csv`

These files are useful as references, but you should regenerate results for any formal comparison or reporting.

## Implementation Notes

### Main source files

- `src/load_model.py`: model loading and device selection
- `src/token_utils.py`: tokenization and target-span alignment
- `src/attention_analysis.py`: attention aggregation metrics
- `src/hidden_analysis.py`: hidden-state pooling and cosine similarity
- `src/run_experiment.py`: end-to-end probe execution
- `src/plot_results.py`: summary plotting

### Attention metrics

For each target span, the probe script aggregates outgoing attention into:

- `self`: a target subword attending to itself
- `intra`: attention from one target subword to another subword of the same target
- `context`: attention from the target span to the rest of the sentence

It also computes incoming attention from the rest of the sentence into the target span.

### Sentence representations

Sentence representations are obtained by mean-pooling hidden states while excluding special tokens when possible. Word representations are obtained by mean-pooling the hidden states over the target token span.

## Reproducibility and Scope

- The repository tracks small curated datasets and lightweight outputs.
- Large raw corpora belong in `data/raw/` and are intentionally excluded from Git.
- Local models and caches belong in `artifacts/models/` and `artifacts/cache/`.
- The current probe set is intentionally small and should be treated as a starting point, not a final benchmark.

## Possible Extensions

Natural next steps for this repository include:

- expanding the verlan pair inventory
- adding more sentence templates and lexical control conditions
- comparing multiple French encoders and causal LMs
- evaluating frequency effects and tokenization robustness
- adding formal statistical summaries on top of the current CSV outputs

## License and Usage

No explicit license file is currently included in the repository. If you plan to reuse or publish this code, add a license and cite the data sources and model checkpoints used in your experiments.
