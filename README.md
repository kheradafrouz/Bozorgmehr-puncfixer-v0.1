<h1 align="center">Bozorgmehr-puncfixer-v0.1</h1>

<p align="center">
  <a href="https://github.com/kheradafrouz/Bozorgmehr-puncfixer-v0.1/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/kheradafrouz/Bozorgmehr-puncfixer-v0.1/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Devices" src="https://img.shields.io/badge/device-CUDA-4ade80">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

Persian punctuation restoration with a fine-tuned ALBERT encoder. Give it Persian text without punctuation and it inserts `،` `.` `؟` `!` `؛` `:` after the right words. It only inserts punctuation and never changes, adds or removes words.

```text
in : سلام حال شما چطور است
out: سلام، حال شما چطور است؟
```

> The example is illustrative. See [Evaluation](#evaluation) for measured results.

## Highlights

- **Token classification, not generation.** The model labels each word with the punctuation that follows it, so the words of the output are the words of the input. This is checked on every call and a `RuntimeError` is raised if it ever fails.
- **Lightweight.** A 44 MB `safetensors` file (ALBERT base). It runs on CPU and uses a GPU automatically if one is available.
- **Long text supported.** Input is split into overlapping windows, so nothing is truncated and no punctuation is forced at window boundaries.
- **Batch and single-string API.** Order is preserved.

## Model details

| | |
|---|---|
| Base model | [`HooshvareLab/albert-fa-zwnj-base-v2`](https://huggingface.co/HooshvareLab/albert-fa-zwnj-base-v2) |
| Task | Token classification (punctuation after each word) |
| Dataset | [`MohammadJRanjbar/persian-punctuation-restoration`](https://huggingface.co/datasets/MohammadJRanjbar/persian-punctuation-restoration) |
| Labels | `NONE`, `COMMA` (`،`), `PERIOD` (`.`), `QUESTION` (`؟`), `EXCLAMATION` (`!`), `SEMICOLON` (`؛`), `COLON` (`:`) |
| Max sequence length | 256 sub-words per window |
| Training scope | subset of the train split: 640,000 sentences (656,000 stream rows read, 3.87% of the split), 20,000 optimizer steps, about 34 minutes on one T4 |
| Hardware | NVIDIA Tesla T4 (Google Colab), fp16 mixed precision |

## Download model

You can obtain the official, ready-to-use version of the model from Hugging Face: [Link](https://huggingface.co/kheradafrouz/Bozorgmehr-puncfixer-v0.1)

## Usage

```python
from punc_inference import PunctuationRestorer

restorer = PunctuationRestorer("path/to/model_folder")   # CPU or GPU is picked automatically

restorer.restore("آیا فردا به دانشگاه می‌روی")            # str -> str
restorer.restore(["جمله اول", "جمله دوم"])                # list -> list, order preserved
```

On Windows, enable UTF-8 so Persian prints correctly:

```powershell
$env:PYTHONUTF8 = "1"
python your_script.py
```

### Constructor options

| Argument | Default | Meaning |
|---|---|---|
| `model_dir` | required | Folder containing the model and tokenizer files |
| `device` | auto | `"cpu"` or `"cuda"` |
| `max_length` | `256` | Sub-word window size |
| `overlap_words` | `16` | Word overlap between consecutive windows for long text |
| `batch_size` | `32` | Windows per forward pass |
| `fp16` | auto | fp16 autocast, on only when running on CUDA |

## Behaviour and design decisions

- **Existing punctuation is discarded and re-predicted.** Input is reduced to words with the same parser used in training, and whitespace is normalised. Numbers such as `3.5`, `10:30` or `1,000` are kept intact.
- **Normalisation.** ASCII `,` and `?` are mapped to `،` and `؟` (except inside numbers).
- **Unsupported symbols** (quotes, brackets, dashes, ellipses) stay inside their word and are never predicted.
- **Excluded from training and evaluation:** sentences with consecutive marks (`؟!`, `...`), leading punctuation, or a token made only of symbols (for example a closing quote after a comma). The model is untested on these patterns.
- **Alignment.** Each word is tokenised separately and its label sits on its last sub-word; all other positions are ignored in the loss.

## Evaluation

All numbers below were measured by the training notebook on the full published test split (327,939 sentences). Sentences that cannot be represented by the label scheme are excluded (1.11% of the evaluated sentences).

| Metric | Model | Baseline (no punctuation) |
|---|---|---|
| Macro F1 | 0.8046 | 0.1337 |
| Micro F1 (punctuation classes only) | 0.8499 | n/a |
| Accuracy (all word positions) | 0.9692 | 0.8797 |
| Insertion precision / recall | 0.9195 / 0.8515 | n/a |
| Punctuation error rate | 0.2560 | 1.0 |
| Full-sentence exact match | 0.6175 | 0.0000 |
| Lexical change rate | 0.0 (0 changes in 324,261 checked sentences; target 0) | n/a |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NONE (no mark) | 0.9799 | 0.9898 | 0.9848 | 5,962,767 |
| COMMA ، | 0.8262 | 0.7228 | 0.7711 | 405,830 |
| PERIOD . | 0.9496 | 0.9841 | 0.9665 | 288,013 |
| QUESTION ؟ | 0.9125 | 0.8344 | 0.8717 | 12,903 |
| EXCLAMATION ! | 0.7126 | 0.4268 | 0.5338 | 23,345 |
| SEMICOLON ؛ | 0.9357 | 0.4688 | 0.6246 | 4,966 |
| COLON : | 0.9216 | 0.8410 | 0.8794 | 80,547 |

Validation split (17,432 sentences, 17,254 usable): macro F1 0.8144, punctuation micro F1 0.8520, accuracy 0.9696, exact match 0.6212. These agree closely with the test numbers. Model selection during training used a 5,000-sentence validation slice, whose macro F1 (0.695) is not comparable to the full-split figures.

Metric definitions:

- Precision, recall and F1 are computed per class over valid word positions.
- Macro F1 averages F1 over classes present in the reference.
- Insertion precision and recall measure whether a mark belongs at a position, ignoring which mark.
- Punctuation error rate is (substitutions + insertions + deletions) divided by the number of reference marks.

Inference speed (Tesla T4, fp16, includes tokenisation): 12.7 (p95 14.9) ms per sentence (median, single input), 487 sentences/s batched.

The no-punctuation baseline has high accuracy only because 88% of word positions carry no mark, so macro F1 and micro F1 over punctuation classes are the informative metrics.

No published score is claimed for this model. Results from other papers use different data and splits and are not directly comparable.

## Error analysis

Based on the test confusion matrix (6,778,371 word positions) and a sample of sentence-level errors:

- **Commas are under-predicted.** About 27% of reference commas receive no mark (recall 0.72, precision 0.83). Comma placement is often a matter of writing style, so some of these misses may be acceptable alternatives.
- **`!` is mostly confused with `.`.** 56% of reference exclamation marks are predicted as periods (recall 0.43). Sentence-final `.` versus `!` often depends on tone, which the text alone does not settle.
- **`؛` is rare and weak.** Recall is 0.47 on 4,966 reference marks, and the model rarely over-predicts it (precision 0.94).
- **Periods, question marks and colons are strong** (F1 0.97, 0.87, 0.88).
- **Dialogue and speaker labels** (`name: utterance`) are a recurring error type, with colon, comma and period swapped.

Examples from the held-out error sample (reference vs. prediction):

| Reference | Prediction |
|---|---|
| مجتبی: بله. | مجتبی: بله! |
| مردم، می‌دانید من کیستم؟ | مردم می‌دانید، من کیستم؟ |
| در اصول عقاید اما، داستان برعکس است. | در اصول عقاید، اما داستان برعکس است. |
| دستهام تره، بگذارید روی میز! | دستهام تره، بگذارید روی میز. |

Many of these are plausible alternatives rather than clear mistakes, so the metrics under-count acceptable output. Only the lexical-change rate is a hard guarantee.

## Limitations

- Trained on written Persian (Wikipedia, blogs, news, Telegram and similar sources) filtered to sentences with proper punctuation. Speech transcripts (ASR output) or very informal text may behave differently.
- Only six marks are predicted. Quotes, brackets, dashes and ellipses are not.
- The model was trained on about 3.9% of the training split in roughly 35 minutes. Rare classes (`!` and `؛`) are the weakest, and further training may help (not tested).
- Exact-string overlap was found between the splits: 66 sentences appear in both validation and test, and 369 of 100,000 sampled training sentences (0.37%) also occur in validation or test, despite the dataset's published deduplication (whitespace normalisation only is applied in this check). Test scores may be slightly optimistic. Only a bounded sample was checked, and the remainder of the train split was not compared.
- Punctuation is partly subjective. Several placements can be acceptable, so the metrics under-count valid alternatives.

## Training

Fine-tuned with a custom resumable loop on a Colab T4 (20,000 steps in two sessions, 0 out-of-memory skips and 0 non-finite losses): AdamW, learning rate 3e-5, weight decay 0.01, 6% linear warm-up then linear decay, effective batch 32 (16 × 2 accumulation), fp16 autocast, gradient clipping 1.0, dynamic padding. The checkpoint with the best validation macro F1 was selected. Validation data was used only for model selection, and the test split only for the final evaluation.

## Repository layout

```text
model.safetensors        fine-tuned weights
config.json              model configuration (id2label / label2id included)
tokenizer.json, tokenizer_config.json
label_mappings.json      label to punctuation mapping
punc_core.py             parsing, labels, alignment, windowing (no dependencies)
punc_inference.py        PunctuationRestorer
SHA256SUMS.json          file checksums
```

## Citation

If you use this model, please also cite the dataset paper and the base model:

```bibtex
@inproceedings{kalahroodi-etal-2026-persianpunc,
  title     = {{P}ersian{P}unc: A Large-Scale Dataset and {BERT}-Based Approach for {P}ersian Punctuation Restoration},
  author    = {Kalahroodi, Mohammad Javad Ranjbar and Faili, Heshaam and Shakery, Azadeh},
  booktitle = {Proceedings of the First Workshop on {NLP} and {LLM}s for the {I}ranian Language Family},
  year      = {2026},
  address   = {Rabat, Morocco},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.silkroadnlp-1.11/}
}

@misc{ALBERTPersian,
  author = {Hooshvare Team},
  title  = {ALBERT-Persian: A Lite BERT for Self-supervised Learning of Language Representations for the Persian Language},
  year   = {2021},
  howpublished = {\url{https://github.com/m3hrdadfi/albert-persian}}
}
```

Also:
```bibtex
@software{bozorgmehr-puncfixer-v0.1,
  title = {Bozorgmehr-puncfixer-v0.1},
  author = {Kheradafrouz},
  url = {https://github.com/kheradafrouz/Bozorgmehr-puncfixer-v0.1},
  license = {apache-2.0}
}
```


## License

Code and weights: [Apache-2.0](LICENSE) (matching the base model). The training data aggregates several source datasets with their own licenses. Check them before commercial use.