# Baby26 Rewrite Pairing

This repository contains the BabyLM 2026 simplification data package and evaluation results for three 10M-token simplification-pair training sets. It also includes the original `train_mask.py` script and the exact notes for running WWM7 -> Token3 with regular MLM.

## Contents

- `data/`: three 10M-token simplification training files.
- `01_fw_simp_v1_sl_10m_40k/`: FineWeb simplification runs and evaluation results.
- `02_simple_wiki_pairs_10m_40k/`: SimpleWiki sentence-pair runs and evaluation results.
- `03_openstax_sentences_bccampus_pairs_10m_40k/`: OpenStax + BCcampus sentence-pair runs and evaluation results.
- `RESULTS_SUMMARY.md`: standalone copy of the project result tables.
- `results_summary.csv`: machine-readable result summary.
- `tokenizers/`: SentencePiece tokenizers used by the three training sets.
- `train_mask.py`: original training script used for regular MLM / WWM curriculum runs.
- `train_mask_wwm_token.py`: clean cut of `train_mask.py` that keeps only ordinary token masking and whole-word masking.

Model checkpoints are intentionally not included in this repository.

## Tokenizers

Each training set has a corresponding tokenizer in `tokenizers/`:

| Training set | Tokenizer |
| --- | --- |
| `fw_simp_v1_sl_10m_40k` | `tokenizers/fw_simp_v1_40k.model` |
| `simple_wiki_pairs_10m_40k` | `tokenizers/simple_wiki_sentences_10m_40k.model` |
| `openstax_sentences_bccampus_pairs_10m_40k` | `tokenizers/openstax_bccampus_original_10m_dedup_40k.model` |

The matching `.vocab` files are included next to each `.model` file.

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

## Folder Layout

Each dataset folder contains `token/eval_results` and `wwm7tok3/eval_results`. The original local archive also contained model checkpoints, but those are excluded here to keep the GitHub repository lightweight.

For the full result tables, see `RESULTS_SUMMARY.md` or `results_summary.csv`.

# `train_mask.py` vs `train_mask_wwm_token.py`

`train_mask.py` is the original full training script. It contains the regular MLM path, whole-word masking, token masking, adaptive masking machinery, mask-weight updates, auxiliary batch fields, and extra masking modes/features used in earlier experiments.

`train_mask_wwm_token.py` is a clean cut made from `train_mask.py`. It keeps the parts needed for:

- ordinary token masking
- whole-word masking
- WWM -> token curriculum, e.g. `--span_masking wwm --span_epochs 7`
- resume from checkpoint
- curriculum switch logging
- final-checkpoint-only training when `--all_checkpoints` is not used

The clean version removes the original AMLM/adaptive-masking path and non-token/non-WWM masking modes. It is meant for controlled regular MLM experiments where the only masking choices are `token` and `wwm`.

For the WWM7 -> Token3 setup used here, the intended command is the same as the `train_mask.py` command above, except the script name can be changed:

```bash
python train_mask_wwm_token.py \
  --train_data data/fw_simp_v1_sl_10m.train \
  --valid_data data/bb26.dev \
  --tokenizer tokenizers/fw_simp_v1_40k.model \
  --model_path configs/deberta-v3-base \
  --output_path output/mlm_wwm7tok3_fw_simp_v1_sl_10m_40k_seed0_clean \
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
  --span_masking wwm \
  --span_epochs 7 \
  --token_epochs 0 \
  --logging_steps 100 \
  --eval_steps 500 \
  --save_steps 999999999 \
  --lamb
```

In this regular MLM setting, `train_mask_wwm_token.py` uses uniform whole-word sampling. The original `train_mask.py` can also run the same WWM7 -> Token3 objective when called with `--regular_mlm`, but it still carries the extra AMLM/adaptive code paths. The clean script removes those paths to make the experiment surface smaller and easier to audit.
