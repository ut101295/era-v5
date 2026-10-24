# raw — HuggingFace hub probes

The actual output of the live probing commands used to select the dataset
(`02-dataset-selection.md` is the interpreted version).

## The example dataset does not exist

```
$ get_dataset_config_names("lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled")
DatasetNotFoundError("Dataset '…' doesn't exist on the Hub or cannot be accessed.")
```

## IndicCorpV2 — languages are splits

```
$ get_dataset_config_names("ai4bharat/IndicCorpV2")
['indiccorp_v2']              # one config

$ load_dataset("ai4bharat/IndicCorpV2", streaming=True, split="train")
ValueError: Bad split: train. Available splits:
['asm_Beng', 'ben_Beng', 'brx_Deva', 'doi_Deva', 'gom_Deva', 'guj_Gujr',
 'hin_Deva', 'kan_Knda', 'kas_Arab', 'mai_Deva', 'mal_Mlym', 'mar_Deva',
 'mni_Mtei', 'npi_Deva', 'ory_Orya', 'pan_Guru', 'san_Deva', 'snd_Deva',
 'tam_Taml', 'tel_Telu', 'urd_Arab', 'khasi', 'santhali']

$ next(iter(load_dataset(..., split="asm_Beng")))
ROW 0 keys= ['text']
  text = 'ঘটনাৰ বিৱৰণি মতে বোকাকাতৰ কৈলাখাতত নিয়মিতভাৱে বহা জুৱাৰ আড্ডাত শুকুৰবাৰে …'
ROW 1 keys= ['text']
  text = ''                    # <-- empty rows are common
```

Note the Assamese sample already contains ৱ (U+09F1) and ৰ (U+09F0) — the
Assamese-specific letters used later to disambiguate from Bengali.

## Throughput / size probe of asm_Beng (5,000 rows)

```
docs=5000 empty=2500 words=105400 chars=693261
avg_words/doc=21.1 avg_chars/doc=138.7   (avg ≈ 42 words per NON-empty doc)
time=5.6s -> 888 docs/s
to hit 12M words need ~569,259 docs, ~10.7 min
```

→ ~50% of rows are empty; short sentence-level web text.

## Conversation candidates

```
$ load_dataset("databricks/databricks-dolly-15k", streaming=True, split="train")
OK keys: ['instruction', 'context', 'response', 'category']

$ get_dataset_config_names("ai4bharat/indic-align")
['Indic_ShareLlama', 'Dolly_T', 'OpenAssistant_T', 'WikiHow', 'IndoWordNet', …]

$ next(iter(load_dataset("ai4bharat/indic-align", "Dolly_T", split="train")))
keys include per-language columns: hin_Deva, ben_Beng, tel_Telu, tam_Taml, …
  hin_Deva = [['मछली की एक प्रजाति कौन सी है? …', 'टोपे']]
```

→ `indic-align / Dolly_T` gives real Indic instruction pairs; the Hindi column keeps the
conversation demo India-first.

## What was actually fetched

See [`fetch-summary.json`](fetch-summary.json):

```json
{ "primary_rows": 111459, "primary_tokens": 10000072,
  "inject_rows": 300, "conv_rows": 2000 }
```
