# Baby26 Rewrite Pairing — core reproduction materials

This branch retains only the training inputs, training code/configuration/tokenizers, and per-model evaluation reports requested for the paper. Intermediate checkpoints, per-example predictions, development history snapshots, and additional analysis tools are excluded from the current tree.

## Training inputs and code

- `data/`: the fixed 10M-scale paper inputs, raw/source controls, BB26 baseline and validation data. Exact bytes, hashes and actual word counts are in `reproduction/TRAIN_INPUTS.json`.
- `tokenizers/`, `configs/`, `preprocessing.py`, `train_mask.py`, `train_mask_wwm_token.py`, `requirements.txt`: the training implementation and required configuration. The saved FAU training dependency file is in `reproduction/environment/` and is not a certification that all historical runs used identical package versions.
- `reproduction/runs.json`: input/tokenizer/seed mapping for the 51 main 10M runs.
- `reproduction/corpus_100m/`: lossless chunks of the exact 100M submission input. Restore with `python reproduction/restore_corpus_100m.py`; the restorer checks all chunk and whole-file SHA-256 hashes.

Inspect a command without starting training:

```bash
python reproduction/train_condition.py --condition fw_simp --masking wwm7tok3 --seed 2 --output output/fw_simp_seed2
# Add --execute to train in a Linux CUDA environment with bitsandbytes LAMB.
# For the 100M submission, restore its corpus, then use --condition submission100m --seed 1.
```

In this implementation `batch_size` is the total batch and `grad_acc` divides it into microbatches; do not multiply them. The 10M setup is hidden/intermediate 384/1280, batch 256, grad_acc 2; 100M is 768/2560, batch 2048, grad_acc 4 and span p=0.5. Sequence length changes 64→256 with inverse batch-size scaling.

## Model reports

`reports/local/` and `reports/fau/` preserve the saved per-model summary reports from the two source locations. They may reflect different evaluation versions; neither is silently substituted for the other. Reports include task results, zero-shot temperature reports and reading correlations. Per-example prediction payloads are not retained in the current tree. `results_summary.csv` and `RESULTS_SUMMARY.md` provide the rebuilt 51-run summary; MRPC and QQP use F1.

Final published models remain on Hugging Face: [10M](https://huggingface.co/go76dof/wwm_curriculum_simplification_40k), [100M](https://huggingface.co/go76dof/spangeo05_simplification). Their file metadata is recorded in `reproduction/PUBLISHED_MODELS.json`. The official evaluation toolkit is [BabyLM eval](https://github.com/babylm-org/babylm-eval).

The exact released inputs take precedence over nominal 10M labels and historical run names. The global BB26 baseline uses the recovered `bb24.model`; explanation and paraphrase inputs contain the OpenStax fill recorded in their original names. Published reports are preserved as evidence, not a new claim that every paper number has been independently reproduced. No training or model scoring was rerun for this release. Original source rights apply to corpora; the code license does not relicense third-party data.
