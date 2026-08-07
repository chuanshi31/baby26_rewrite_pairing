# Baby26 Simplification

This repository contains the BabyLM 2026 simplification data package and evaluation results for three 10M-token simplification-pair training sets. It also includes the original `train_mask.py` script and the exact notes for running WWM7 -> Token3 with regular MLM.

## Contents

- `data/`: three 10M-token simplification training files.
- `01_fw_simp_v1_sl_10m_40k/`: FineWeb simplification runs and evaluation results.
- `02_simple_wiki_pairs_10m_40k/`: SimpleWiki sentence-pair runs and evaluation results.
- `03_openstax_sentences_bccampus_pairs_10m_40k/`: OpenStax + BCcampus sentence-pair runs and evaluation results.
- `RESULTS_SUMMARY.md`: standalone copy of the project result tables.
- `results_summary.csv`: machine-readable result summary.
- `train_mask.py`: original training script used for regular MLM / WWM curriculum runs.

Model checkpoints are intentionally not included in this repository.

# Using `train_mask.py` for WWM7 -> Token3

This note describes how to run the original `train_mask.py` in the plain MLM setting with a whole-word-masking curriculum: WWM for the first 7 epochs, then ordinary token masking for the final 3 epochs.

## Core Idea

Use:

```bash
--regular_mlm
--span_masking wwm
--span_epochs 7
--token_epochs 0
--epochs 10
```

This means:

- epochs `0-6`: use `wwm`
- epochs `7-9`: switch to `token`
- total: 10 epochs

The `--regular_mlm` flag is important. It disables the adaptive masking weight updates in `train_mask.py`, so the run stays in the normal MLM path while still using the selected masking mode.

## Example Command

```bash
python train_mask.py \
  --train_data data/fw_simp_v1_sl_10m.train \
  --valid_data data/bb26.dev \
  --tokenizer tokenizers/fw_simp_v1_40k.model \
  --model_path configs/deberta-v3-base \
  --output_path output/mlm_wwm7tok3_fw_simp_v1_sl_10m_40k_seed0 \
  --hidden_size 384 \
  --intermediate_size 1280 \
  --dropout 0.1 \
  --max_seq_len 0:64,5:256 \
  --batch_size 256 \
  --grad_acc 2 \
  --lr 0.007 \
  --epochs 10 \
  --weight_decay 0.01 \
  --seed 0 \
  --cpus 8 \
  --mlm_prob 0.15 \
  --mask_decay 0 \
  --mask_replace_prob 0.8 \
  --random_replace_prob 0.1 \
  --mask_update_steps 500 \
  --first_mask_update 0 \
  --span_masking wwm \
  --span_epochs 7 \
  --token_epochs 0 \
  --span_stats_mode average \
  --logging_steps 100 \
  --eval_steps 500 \
  --save_steps 999999999 \
  --regular_mlm \
  --lamb
```

## Checkpoint Behavior

To keep only the final checkpoint, do not pass:

```bash
--all_checkpoints
```

Use a very large save interval instead:

```bash
--save_steps 999999999
```

`train_mask.py` still saves a final checkpoint at the end of training:

```text
checkpoint-<total_steps>
```

## Important Parameters

`--span_masking wwm`  
Selects whole-word masking as the initial masking strategy.

`--span_epochs 7`  
Uses `--span_masking` for the first 7 epochs, then switches to token masking.

`--token_epochs 0`  
Keeps the curriculum direction as WWM first, token second. Do not set both `--span_epochs` and `--token_epochs` for this setup.

`--regular_mlm`  
Makes `train_mask.py` behave like regular MLM. This prevents AMLM mask-weight updates from changing the masking distribution.

`--mlm_prob 0.15`  
Masks approximately 15% of tokens. In WWM mode, whole words are selected until the masked token count reaches this target.

`--mask_replace_prob 0.8` and `--random_replace_prob 0.1`  
Use the standard BERT-style corruption rule: 80% `[MASK]`, 10% random token, 10% unchanged.

`--max_seq_len 0:64,5:256`  
Uses sequence length 64 for early epochs, then 256 starting at epoch 5.

## Expected Masking Schedule

| Epoch | Masking mode |
|---:|---|
| 0 | WWM |
| 1 | WWM |
| 2 | WWM |
| 3 | WWM |
| 4 | WWM |
| 5 | WWM |
| 6 | WWM |
| 7 | Token |
| 8 | Token |
| 9 | Token |

## Minimal Checklist

- `--regular_mlm` is present
- `--span_masking wwm` is present
- `--span_epochs 7` is present
- `--epochs 10` is present
- `--all_checkpoints` is absent if only the final checkpoint should be kept
- `--save_steps 999999999` is used if intermediate checkpoints should be avoided

# Project Report

Scores are extracted from the eval reports included in this folder. Entity uses the updated Entity Tracking report, matching the later update table in `SEED_RESULTS.md`.

## Per-Seed Results

| Dataset | Masking | Seed | Run | BLiMP | Supp | EWoK | Entity Updated | COMPS | Eye | SPR | ZS Avg7 | GlobalPIQA Avg | FT Avg7 | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FineWeb Simp V1 | token | 0 | mlm_token_fw_simp_v1_sl_10m_40k | 65.97 | 55.54 | 52.64 | 23.48 | 52.39 | 6.60 | 3.91 | 37.22 | 36.13 | 68.48 | 52.85 |
| FineWeb Simp V1 | token | 1 | mlm_token_fw_simp_v1_sl_10m_40k_seed1_v100 | 65.24 | 51.82 | 53.28 | 27.79 | 51.53 | 6.96 | 3.93 | 37.22 | 38.65 | 68.22 | 52.72 |
| FineWeb Simp V1 | token | 2 | mlm_token_fw_simp_v1_sl_10m_40k_seed2_v100 | 65.86 | 54.99 | 52.75 | 27.07 | 52.07 | 4.33 | 2.42 | 37.07 | 36.70 | 68.32 | 52.69 |
| FineWeb Simp V1 | wwm7tok3 | 0 | mlm_wwm7tok3_fw_simp_v1_sl_10m_40k_seed0_v100 | 67.54 | 54.19 | 55.13 | 29.12 | 52.71 | 7.37 | 4.36 | 38.63 | 35.73 | 69.83 | 54.23 |
| FineWeb Simp V1 | wwm7tok3 | 1 | mlm_wwm7tok3_fw_simp_v1_sl_10m_40k_seed1_v100 | 68.02 | 54.44 | 54.90 | 29.13 | 52.14 | 6.54 | 4.28 | 38.49 | 40.61 | 69.96 | 54.22 |
| FineWeb Simp V1 | wwm7tok3 | 2 | mlm_wwm7tok3_fw_simp_v1_sl_10m_40k_seed2_v100 | 67.20 | 56.01 | 56.07 | 28.45 | 53.57 | 6.83 | 4.02 | 38.88 | 39.66 | 69.79 | 54.34 |
| SimpleWiki Pairs | token | 0 | mlm_token_simple_wiki_pairs_10m_40k_seed0_v100 | 66.20 | 48.74 | 51.65 | 23.55 | 52.08 | 7.34 | 2.93 | 36.07 | 38.09 | 69.57 | 52.82 |
| SimpleWiki Pairs | token | 1 | mlm_token_simple_wiki_pairs_10m_40k_seed1_v100 | 66.26 | 46.87 | 51.65 | 24.92 | 52.20 | 7.14 | 2.79 | 35.98 | 33.23 | 68.65 | 52.31 |
| SimpleWiki Pairs | token | 2 | mlm_token_simple_wiki_pairs_10m_40k_seed2_v100 | 65.72 | 49.05 | 50.96 | 21.23 | 51.78 | 5.22 | 2.51 | 35.21 | 36.66 | 67.92 | 51.56 |
| SimpleWiki Pairs | wwm7tok3 | 0 | mlm_wwm7tok3_simple_wiki_pairs_10m_40k_seed0_v100 | 66.27 | 48.66 | 54.03 | 25.15 | 52.35 | 8.20 | 2.93 | 36.80 | 32.68 | 70.32 | 53.56 |
| SimpleWiki Pairs | wwm7tok3 | 1 | mlm_wwm7tok3_simple_wiki_pairs_10m_40k_seed1_v100 | 66.13 | 49.04 | 53.92 | 21.72 | 52.34 | 7.62 | 3.39 | 36.31 | 34.18 | 69.77 | 53.04 |
| SimpleWiki Pairs | wwm7tok3 | 2 | mlm_wwm7tok3_simple_wiki_pairs_10m_40k_seed2_v100 | 66.31 | 49.33 | 52.25 | 24.46 | 52.85 | 6.97 | 2.71 | 36.41 | 37.15 | 69.41 | 52.91 |
| OpenStax+BCCampus Pairs | token | 0 | mlm_token_openstax_sentences_bccampus_pairs_10m_40k_seed0_v100 | 67.46 | 58.82 | 52.25 | 25.34 | 51.01 | 6.22 | 3.96 | 37.87 | 41.56 | 69.81 | 53.84 |
| OpenStax+BCCampus Pairs | token | 1 | mlm_token_openstax_sentences_bccampus_pairs_10m_40k_seed1_v100 | 66.81 | 55.57 | 53.15 | 23.05 | 51.00 | 7.41 | 3.57 | 37.22 | 38.53 | 68.82 | 53.02 |
| OpenStax+BCCampus Pairs | token | 2 | mlm_token_openstax_sentences_bccampus_pairs_10m_40k_seed2_v100 | 67.45 | 56.32 | 55.61 | 27.95 | 51.01 | 5.02 | 3.58 | 38.13 | 41.08 | 69.04 | 53.59 |
| OpenStax+BCCampus Pairs | wwm7tok3 | 0 | mlm_wwm7tok3_openstax_sentences_bccampus_pairs_10m_40k_seed0_v100 | 67.78 | 58.35 | 54.57 | 25.26 | 51.51 | 7.83 | 3.97 | 38.47 | 38.09 | 68.88 | 53.67 |
| OpenStax+BCCampus Pairs | wwm7tok3 | 1 | mlm_wwm7tok3_openstax_sentences_bccampus_pairs_10m_40k_seed1_v100 | 67.72 | 56.54 | 55.71 | 23.49 | 51.36 | 7.88 | 4.59 | 38.18 | 39.08 | 69.00 | 53.59 |
| OpenStax+BCCampus Pairs | wwm7tok3 | 2 | mlm_wwm7tok3_openstax_sentences_bccampus_pairs_10m_40k_seed2_v100 | 67.51 | 57.59 | 54.40 | 28.16 | 51.26 | 8.11 | 4.13 | 38.74 | 38.18 | 69.52 | 54.13 |

## Three-Seed Means

| Dataset | Masking | BLiMP | Supp | EWoK | Entity Updated | COMPS | Eye | SPR | ZS Avg7 | GlobalPIQA Avg | FT Avg7 | Combined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FineWeb Simp V1 | token | 65.69 | 54.12 | 52.89 | 26.11 | 52.00 | 5.96 | 3.42 | 37.17 | 37.16 | 68.34 | 52.75 |
| FineWeb Simp V1 | wwm7tok3 | 67.59 | 54.88 | 55.37 | 28.90 | 52.81 | 6.91 | 4.22 | 38.67 | 38.67 | 69.86 | 54.26 |
| SimpleWiki Pairs | token | 66.06 | 48.22 | 51.42 | 23.23 | 52.02 | 6.57 | 2.74 | 35.75 | 35.99 | 68.71 | 52.23 |
| SimpleWiki Pairs | wwm7tok3 | 66.24 | 49.01 | 53.40 | 23.78 | 52.51 | 7.60 | 3.01 | 36.51 | 34.67 | 69.83 | 53.17 |
| OpenStax+BCCampus Pairs | token | 67.24 | 56.90 | 53.67 | 25.45 | 51.01 | 6.22 | 3.70 | 37.74 | 40.39 | 69.22 | 53.48 |
| OpenStax+BCCampus Pairs | wwm7tok3 | 67.67 | 57.49 | 54.89 | 25.64 | 51.38 | 7.94 | 4.23 | 38.46 | 38.45 | 69.13 | 53.80 |

## Folder Layout

Each dataset folder contains `token/eval_results` and `wwm7tok3/eval_results`. The original local archive also contained model checkpoints, but those are excluded here to keep the GitHub repository lightweight.
