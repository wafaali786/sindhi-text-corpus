# Sindhi Text Corpus and Continued Pretraining of XLM-RoBERTa

A cleaned Sindhi text corpus (410 MB, 49.2 million words) and the scripts used to build it and to test that it helps a language model.

**Dataset on Hugging Face:** https://huggingface.co/datasets/wafaali786/sindhi-text-corpus

## Why

Sindhi is a low-resource language: there is very little clean, ready-to-use text for training language models. This project cleans two public sources into one corpus, then trains XLM-RoBERTa base on part of it to check that the corpus makes the model better at Sindhi.

No new text was scraped. The corpus is made from two existing public sources, cleaned, de-duplicated and split.

## Results

Perplexity and word-guessing accuracy on a held-out test set (same random masks for both models):

| Model | Test loss | Perplexity | Top-1 accuracy | Top-5 accuracy |
|---|---|---|---|---|
| Original XLM-RoBERTa base | 2.4705 | 11.83 | 47.0% | 70.7% |
| After training on this corpus | 1.9979 | 7.37 | 58.9% | 80.5% |

Accuracy is measured on 1,000 sentences with one hidden word each, using only words that are a single token in XLM-R's vocabulary. Training: 3,000 steps on the first 400,000 training lines, free Google Colab T4 GPU, about 2 hours 21 minutes.

Tokens per word on validation text (lower is better):

| Tokenizer | Tokens per word |
|---|---|
| BPE, 16,000 pieces | 1.23 |
| Unigram, 16,000 pieces | 1.23 |
| BPE, 32,000 pieces | 1.18 |
| Unigram, 32,000 pieces | 1.18 |
| XLM-R original | 1.53 |

## The corpus

| Split | Lines | Words |
|---|---|---|
| Train | 1,016,307 | 48,245,742 |
| Validation | 10,000 | 479,094 |
| Test | 10,000 | 475,594 |
| **Total** | **1,036,307** | **49,200,430** |

Sources: CC-100 Sindhi (967,904 lines kept) and Sindhi Wikipedia (68,403 lines kept).

Cleaning steps: Unicode normalization (NFKC); removal of URLs, email addresses, invisible characters and tatweel; removal of a trailing English site name; dropping lines under 4 words; dropping lines with under 80% Arabic-script letters; removing exact duplicate lines. This removed 627,727 of the 1,664,034 lines read.

The data files are not in this repository because of their size. Load them from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("wafaali786/sindhi-text-corpus")
print(ds)
```

## Scripts

Each script is written for Google Colab. A `# %%` line marks a new cell: paste each block into its own cell and run in order. Files are saved to a `sindhi_corpus` folder in Google Drive.

| Script | What it does | Runtime |
|---|---|---|
| `scripts/step1_download_sindhi_data.py` | Downloads CC-100 Sindhi and Sindhi Wikipedia | CPU |
| `scripts/step2_clean_corpus.py` | Cleans and de-duplicates the text | CPU |
| `scripts/step3_split_and_tokenizer.py` | Splits into train/validation/test and trains four SentencePiece tokenizers | CPU |
| `scripts/step4_continued_pretraining.py` | Baseline perplexity, continued pretraining, before/after perplexity | T4 GPU |
| `scripts/step5_demo.py` | Hidden-word demo on eight sentences | T4 GPU |
| `scripts/step6_word_accuracy.py` | Top-1 and top-5 accuracy on 1,000 questions | T4 GPU |
| `scripts/step7_upload_huggingface.py` | Uploads the splits to Hugging Face (set `USERNAME` first) | CPU |

Random seeds: 42 for the split, training and masking; 7 for the accuracy questions; 1 for the eight-sentence demo.

## Limitations

- The split is random by line, so lines from the same article can appear in both train and test. XLM-R also saw CC-100 text in its original training. A test on freshly collected text would be a stronger check.
- The model trained on 400,000 of the 1,016,307 training lines, in a single run, so there are no error bars.
- Only single-token words were hidden in the accuracy test, which is an easier subset.
- Some site names remain at the end of lines (for example "_ KTN NEWS"). The script filter does not separate Sindhi from other languages that use the same script, and no language-identification model was used.
- The text is mostly news, web pages and Wikipedia. It was not screened for personal or offensive content.

## Data sources and terms

- Sindhi Wikipedia text is available under CC BY-SA 4.0 (https://dumps.wikimedia.org/).
- CC-100 is derived from Common Crawl web pages; copyright in the original pages stays with the website owners, and use follows the Common Crawl terms of use. Please cite Conneau et al. (2020), "Unsupervised Cross-lingual Representation Learning at Scale", and Wenzek et al. (2020), "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data".

The corpus is provided for research and educational use. If you own text in it and want it removed, please open an issue in this repository.

## Author

Wafa Ali - GitHub [@wafaali786](https://github.com/wafaali786) - Hugging Face [wafaali786](https://huggingface.co/wafaali786)
