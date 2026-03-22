# llm_verlan
basic dataset: verlan_base_forms.csv
after manuel selection:velan_database.csv

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