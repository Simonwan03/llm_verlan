# llm_verlan

French verlan data collection and embedding experiments. The repository is currently notebook-first, so the structure is organized around reproducible datasets plus local experiment artifacts.

## Recommended layout

```text
llm_verlan/
├── README.md
├── .gitignore
├── notebooks/                  # notebooks to keep in Git
│   ├── read_data.ipynb
│   ├── train_fasttext.ipynb
│   ├── word_embedding.ipynb
│   └── openai_embedding.ipynb
├── data/
│   ├── raw/                    # local downloads only, do not upload
│   ├── interim/                # generated intermediate files, usually not tracked
│   └── processed/              # small curated datasets, keep in Git
├── artifacts/
│   ├── models/                 # pretrained / trained model binaries, local only
│   ├── cache/                  # API or embedding cache, local only
│   └── figures/                # generated plots, local only by default
└── outputs/                    # evaluation tables / ad hoc text outputs, local only
```

## What should be maintained in Git and uploaded to GitHub

- `README.md`, `.gitignore`
- everything under `notebooks/`
- small, human-curated datasets under `data/processed/`
- future lightweight source code such as `src/`, `scripts/`, or config files if you add them later

For this repository, the files that are good candidates to keep in Git are:

- `data/processed/verlan_base_forms.csv`
- `data/processed/verlan_database.csv`
- `data/processed/wiktionary_verlan_titles.csv`

These files are small, important to the project, and useful for collaborators to reproduce your experiments quickly.

## What should stay local and not be uploaded

- `data/raw/`
  Raw downloads such as OPUS / OpenSubtitles dumps and Kaikki or Wiktionary extracts.
- `data/interim/`
  Rebuildable files produced from raw data cleaning or filtering.
- `artifacts/models/`
  Large binaries such as `cc.fr.300.bin`, `cc.fr.300.bin.gz`, or your trained fastText models.
- `artifacts/cache/`
  OpenAI embedding cache or any other cache files.
- `artifacts/figures/`
  Generated plots unless you explicitly want one figure versioned for a paper/report.
- `outputs/`
  Retrieval tables, neighbor dumps, sentence probes, and other experiment outputs.

The rule of thumb is simple:

- keep small inputs and hand-maintained reference data
- do not keep large downloads, generated binaries, caches, or one-off outputs

## Why this split is better

- The Git history stays small and usable on GitHub.
- Collaborators can immediately see what is source data, what is curated data, and what is generated output.
- Your notebooks become easier to reuse because all paths now follow a consistent convention.
- It becomes straightforward to later add `src/` or `scripts/` without mixing code and data artifacts in the repository root.

## Notebook workflow

1. Put downloaded corpora and dictionary dumps into `data/raw/`.
2. Run `notebooks/read_data.ipynb` to build filtered or curated datasets.
3. Save reusable small outputs into `data/processed/`.
4. Save large models to `artifacts/models/`.
5. Save caches, plots, and experiment outputs to `artifacts/` or `outputs/`.

Before running the embedding notebooks, make sure required local assets are placed in the expected folders. For example, `word_embedding.ipynb` expects the pretrained fastText file at `artifacts/models/cc.fr.300.bin`.

## Current data / experiment notes

- `read_data.ipynb` prepares verlan candidate data from Wiktionary or Kaikki-derived sources.
- `train_fasttext.ipynb` trains a local fastText model on subtitle corpora.
- `word_embedding.ipynb` compares verlan/base pairs with local embedding models.
- `openai_embedding.ipynb` evaluates OpenAI embeddings and writes result tables locally.

## GitHub note

Do not push multi-GB files such as subtitle corpora or fastText binary models into normal Git history. If you ever need to publish a medium or large artifact, use GitHub Releases, external storage, or Git LFS instead of regular commits.
