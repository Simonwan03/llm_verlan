# llm_verlan


# datasets
wikitionary(https://en.wiktionary.org/wiki/Category:Verlan) includes 312 relevant pages of verlan, also kaikki(https://kaikki.org/) has already archived the raw and preprocessed of all the words from wikitionary(in .csv format which is easy to use and process ourselves).

For real life scenarios, we use the datasets from https://opus.nlpl.eu (https://opus.nlpl.eu/datasets/OpenSubtitles?pair=fr&en), which is a collecetion of OpenSubtitles(including the subtitles from all kinds of streaming). And then we extract the sentences containing verlans.

# experiments
## jellyfish 
the link of the documents: https://github.com/jamesturk/jellyfish

`jellyfish` is a Python library for **string matching**. It is mainly used for:

- **edit distance / string similarity**
- **phonetic encoding** for words that sound similar

It is useful for tasks like:

- typo correction
- fuzzy matching
- deduplication
- record linkage
- name matching

### Common functions

- `levenshtein_distance()`  
  Computes the minimum number of single-character edits needed to change one string into another.

- `damerau_levenshtein_distance()`  
  Similar to Levenshtein distance, but also allows character transpositions.

- `jaro_similarity()`  
  Measures similarity between two strings, often useful for short strings like names.

- `jaro_winkler_similarity()`  
  A variant of Jaro that gives more weight to matching prefixes.

- `hamming_distance()`  
  Counts how many positions differ between two strings of equal length.

- `soundex()`, `metaphone()`, `nysiis()`, `match_rating_codex()`  
  Phonetic encoding methods for comparing words by pronunciation.

## SpaCy
The link of official documents: https://spacy.io/models/fr#fr_core_news_lg

text
→ tokenizer
→ tok2vec
→ morphologizer
→ parser
→ attribute_ruler
→ lemmatizer
→ ner
→ annotated Doc

So the flow is roughly:

- tokenize the text
- compute token vectors
- predict morphology / POS
- predict syntax
- apply rule-based attribute fixes
- compute lemmas
- detect named entities

## Word2Vec
## Fasttext
There has been open-source pretrained models:
https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md

We distribute pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia using fastText. These models were trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives. We also distribute three new word analogy datasets, for French, Hindi and Polish.

However, the pretrained datasets normally contains only the formal language materials, but not the informal ones like verlans. So we can actually train a fasttext embedding ourselves using the opensubtitles dataset.

**To do**

## BERT(Camembert)
https://huggingface.co/docs/transformers/model_doc/camembert

a BERT type transformer model training on a french dataset

We can use it to see 

## Chatgpt embedding 


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

那就直接按一个**能写进论文、也能真正跑出来**的方案来做。

---

# 一、核心问题

你现在最值得问的问题不是：

> “这张 attention 图好不好看？”

而是：

> **和标准词相比，verlan 在模型内部是否表现出更强的碎片化处理，以及这种处理是否会随着层数演化为更全局的语义整合？**

拿 `meufs` vs `femmes` 做例子，这个问题就很清楚。

---

# 二、实验主线

建议你把实验分成三层：

## 1. Tokenization 层

看 verlan 是否更容易被拆碎。

你要统计：

* token 个数
* 一个目标词被拆成几个 subword
* 是否出现很短的碎片 token
* verlan 和标准词的 tokenization 差异

例如：

* `femmes` → `_femmes`
* `meufs` → `_me`, `uf`, `s`

这一步是最基础证据。

---

## 2. Layer-wise attention 层

看模型是否在浅层更关注 verlan 内部碎片，在中高层再转向上下文。

这一层不要只看热图，要做**定量指标**。

---

## 3. Representation / understanding 层

看 `meufs` 和 `femmes` 在上下文里的表示是否接近。

这一步比 attention 更重要。你可以做：

* layer-wise hidden state cosine similarity
* sentence embedding cosine similarity
* MLM probability 或 pseudo-perplexity 比较
* 替换前后句子的表示漂移

---

# 三、最重要的 attention 定量指标

下面这几个指标最有用。

---

## 指标 1：词内注意力比例 Intra-word Attention

设 verlan 词被拆成 token 集合 (W)。

例如 `meufs = {_me, uf, s}`。

定义：

[
\text{IntraWord}(l)
===================

\frac{\sum_{h}\sum_{i\in W}\sum_{j\in W, j\neq i} A^{(l,h)}*{ij}}
{\sum*{h}\sum_{i\in W}\sum_{j} A^{(l,h)}_{ij}}
]

含义：

* 在第 (l) 层里
* 这个词的 token 把多少注意力分配给“同一个词内部的其他碎片”

如果 verlan 的这个值在浅层显著更高，就说明模型在做**局部重组**。

---

## 指标 2：上下文注意力比例 Context Attention

定义：

[
\text{Context}(l)
=================

\frac{\sum_{h}\sum_{i\in W}\sum_{j\notin W} A^{(l,h)}*{ij}}
{\sum*{h}\sum_{i\in W}\sum_{j} A^{(l,h)}_{ij}}
]

含义：

* 目标词 token 把多少注意力投向句子其他部分

如果你看到：

* 浅层 `IntraWord` 高
* 中高层 `Context` 上升

那就很支持“先局部处理、后全句整合”的假说。

---

## 指标 3：被关注强度 Incoming Attention

定义：

[
\text{Incoming}(l)
==================

\frac{\sum_{h}\sum_{i\notin W}\sum_{j\in W} A^{(l,h)}*{ij}}
{\sum*{h}\sum_{i}\sum_{j\in W} A^{(l,h)}_{ij}}
]

含义：

* 句子里其他词是否更关注这个 verlan 词

这个可以帮你看：
verlan 是否在句内成为一个“需要被解释”的特殊点。

---

## 指标 4：attention entropy

对目标词 token 的注意力分布算熵：

[
H(l) = -\sum_j p_j \log p_j
]

如果 verlan 的 attention entropy 更高，说明它的注意力更分散；
如果更低但集中在内部碎片上，说明它更局部化。

---

# 四、必须做标准对照

这个特别重要。
你不能只分析 `meufs`，必须同时分析 `femmes`。

比如固定句子模板：

* `Les femmes sont très joyeuses.`
* `Les meufs sont très joyeuses.`

然后逐层比较：

* tokenization
* intra-word attention
* context attention
* hidden-state similarity

否则你永远不知道你看到的是：

* verlan 特有现象
  还是
* 普通 subword 词都会有的现象

---

# 五、建议的数据设计

不要只做一个词。至少做一个小数据集。

## 词表分组

你可以选 20–50 个 verlan / standard pairs，按三类分组：

### A. 高频、已词汇化

* meuf / femme
* teuf / fête
* reum / mère
* keuf / flic ? 不一定一一对应，谨慎

### B. 形式上像 verlan，但模型可能不熟

* vénère / énervé
* zarbi / bizarre
* ouf / fou

### C. 多词或更复杂形式

* laisse béton / laisser tomber
* ça comme / comme ça

这样你能比较不同类型。

---

## 句子模板

尽量控制上下文，只替换目标词。

比如每个 pair 用 3–5 个模板：

* `Cette X est sympathique.`
* `Je parle avec cette X.`
* `Il pense à sa X.`
* `On voit les X dans la rue.`

这样你可以减少句法和语境干扰。

---

# 六、representation 分析比 attention 更关键

这部分强烈建议做。

## 1. 目标词 hidden state 相似度

对于每一层 (l)，取目标词所在 token 的 hidden states。

如果目标词拆成多个 token，就可以：

* 平均池化
* 或取第一个 token
* 或做 attention-weighted pooling

然后比较：

[
\cos(\mathbf{h}^{(l)}*{\text{meufs}}, \mathbf{h}^{(l)}*{\text{femmes}})
]

如果你发现：

* 浅层相似度低
* 中高层相似度升高

这就很漂亮。它直接对应“逐步语义对齐”。

---

## 2. 整句表示相似度

取 `[CLS]` 或句子平均表示，比较：

[
\cos(\mathbf{s}^{(l)}*{\text{meufs sentence}}, \mathbf{s}^{(l)}*{\text{femmes sentence}})
]

如果整句相似度一直很高，说明模型对句意总体还行；
如果目标词相似度低但整句高，说明模型可能靠上下文补偿。

这个很有意思。

---

## 3. 词替换漂移

定义：

[
\Delta^{(l)} = 1 - \cos(\mathbf{s}^{(l)}*{\text{verlan}}, \mathbf{s}^{(l)}*{\text{standard}})
]

这个就是“verlan 替换带来的表示漂移”。

---

# 七、你可以提出的假说

可以提前写成 3 个 hypothesis。

## H1

Verlan forms undergo more fragmented tokenization than their standard counterparts.

## H2

Early transformer layers allocate more attention to intra-word subword reconstruction for verlan forms than for standard forms.

## H3

Despite early fragmentation, hidden representations of verlan forms progressively align with those of their standard counterparts in higher layers.

这三个就构成一条完整故事线：

* 先碎裂
* 再局部重组
* 再逐步语义对齐

---

# 八、结果图应该怎么画

建议别再只放大热图了，改成下面这些。

## 图 1：token 数比较

柱状图：

* standard vs verlan 平均目标词 token 数
* 或平均句长 token 数

---

## 图 2：IntraWord / Context 随层变化

横轴 layer，纵轴指标值。
画两条线：

* verlan
* standard

这会比热图强太多。

---

## 图 3：layer-wise hidden-state similarity

横轴 layer，纵轴 cosine similarity。

看 `meufs` 和 `femmes` 是否随着层数越来越接近。

---

## 图 4：个例热图

最后再放 1–2 个 case study 热图。
这时热图就是辅助说明，不再是主证据。

---

# 九、一个最小可行实验

如果你现在想快速出第一版结果，不要一开始做 50 个词。
先做一个小 MVP：

## 先做 5 对词

* meuf / femme
* teuf / fête
* vénère / énervé
* zarbi / bizarre
* ouf / fou

## 每对 3 个模板

共 15 对句子。

## 跑 1 个模型

比如 CamemBERT-base。

## 先出 4 类结果

* tokenization 对比
* layer-wise intra-word attention
* layer-wise context attention
* layer-wise hidden-state cosine

只要这四个能跑出来，你的项目就已经很像论文了。

---

# 十、一个很实用的判断标准

最后你可以用这个来判断结果是否有价值。

如果你看到的是：

* verlan tokenization 更碎
* 浅层 intra-word attention 更高
* 中层 context attention 上升
* 高层与标准词 hidden states 更接近

那你的结论就会非常顺：

> Verlan forms are initially processed as fragmented subword material, but progressively integrated into more abstract sentence-level representations, suggesting partial internal normalization by the model.

如果最后发现：

* 高层也对不齐
* 句子表示也明显漂移
* MLM 表现也差

那你的结论就变成：

> The model does not robustly normalize verlan forms, and the mismatch persists beyond early tokenization effects.

这两种结果都能写。

---

# 十一、你下一步最该做什么

最推荐的顺序是：

1. 先做 `meufs vs femmes` 的 tokenization 和 layer-wise hidden-state similarity
2. 再做 intra-word / context attention 的逐层统计
3. 最后扩展到 5–20 对 verlan 词

---

如果你愿意，我下一条可以直接给你一份**可运行的 Python 实验代码框架**，用 Hugging Face 一次性把 `meufs vs femmes` 的这些指标全部算出来。

---

# 十二、当前仓库里的最小可运行实现

仓库现在已经补了一套独立脚本，和现有 notebook 分开：

```text
src/
├── load_model.py
├── token_utils.py
├── hidden_analysis.py
├── attention_analysis.py
├── run_experiment.py
└── plot_results.py

scripts/
├── run_verlan_probe.py
└── plot_verlan_probe.py

data/processed/
├── verlan_probe_pairs.csv
└── verlan_probe_templates.csv
```

## 安装依赖

```bash
pip install -r requirements.txt
```

需要的核心包是：

- `torch`
- `transformers`
- `sentencepiece`
- `pandas`
- `matplotlib`

## 运行最小实验

默认会读取 5 对 probe pairs 和按词类分组的模板，然后输出 tokenization、layer-wise similarity、attention 指标。

```bash
python scripts/run_verlan_probe.py --model-name camembert-base
```

结果会写到：

- `outputs/verlan_probe/experiment_results.csv`
- `outputs/verlan_probe/tokenization_results.csv`
- `outputs/verlan_probe/failures.csv`
- `outputs/verlan_probe/run_metadata.json`

## 生成图

```bash
python scripts/plot_verlan_probe.py
```

图和 subword 汇总表会写到：

- `outputs/verlan_probe/figures/`

## 说明

- 目标词 span 不是靠 subword 子串匹配，而是先找字符级词边界，再映射回 tokenizer offsets，所以对 SentencePiece 更稳。
- `experiment_results.csv` 的 `layer=0` 是 embedding output，不是第一层 transformer；attention 指标只从 `layer>=1` 开始有效。
- 默认模板按 `noun_plural`、`noun_singular`、`adj_person`、`adj_thing` 分组，避免把 `teuf`、`vénère`、`zarbi`、`ouf` 硬塞进同一种句法槽位。
