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

To keep only the final checkpoint, do **not** pass:

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

With:

```bash
--epochs 10
--span_masking wwm
--span_epochs 7
--token_epochs 0
```

the schedule is:

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

Before running, check:

- `--regular_mlm` is present
- `--span_masking wwm` is present
- `--span_epochs 7` is present
- `--epochs 10` is present
- `--all_checkpoints` is absent if only the final checkpoint should be kept
- `--save_steps 999999999` is used if intermediate checkpoints should be avoided
