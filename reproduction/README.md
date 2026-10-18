# Reproducing the paper

This release preserves the fixed inputs and evidence needed for the paper, rather than the entire development history or intermediate checkpoints.

## Scope and mapping

- `runs.json`: all 17 10M conditions × 3 seeds = 51 runs. Existing simplification results retain their original directory paths; the other runs are under `results/`.
- Conditions: `fw_raw`, `fw_simp`, `fw_expl`, `fw_para`, `os_raw`, `os_simp`, `sw_raw`, `sw_simp`, and global `bb26` (Token only). Other conditions have Token and WWM7→Token3.
- `results/mlm_spangeo05full_fineweb_os_bcc_100m_40k_seed1_fau`: the 100M submission seed's evaluation evidence. The separate 100M parameter sweeps are outside the paper's required release.
- `raw_wwm_finetune_summary.json`: exact numerical fields of the 63 saved FAU raw-WWM GLUE reports, with source paths and whole-file SHA-256. They complement the locally retained zero-shot results; no missing task is silently averaged away.
- `generated_summary.json`: rebuilt per-seed metrics, condition means and sample standard deviations. MRPC and QQP use F1; other GLUE tasks use accuracy. EWoK subdomain and concept/variable-swap sections are preserved.

The training corpora are **10 million words**, not 10 million tokenizer tokens. Some corpora end at a complete pair and slightly exceed the nominal budget; exact bytes/hashes define the released inputs.

Names are historical: the condition called “FineWeb Explanation” uses the supplied FineWeb explanation corpus with OpenStax fill (see its source manifest), and the paraphrase-only condition uses `fw_os_para_pure_10m.train` with the recorded OpenStax fill. The older `fw_para_neg_pure` experiment contains negation. The paper’s Entity values match that older series rather than the saved paraphrase-only reports, although EWoK/GLUE means match the paraphrase-only series. The reason is unresolved; see `PAPER_NUMERICAL_AUDIT.json`. Its six seed runs, tokenizer and training file are retained as necessary audit evidence. The global BB26 baseline's recovered tokenizer is `bb24.model`; its SHA matches the saved global-baseline checkpoint. Do not infer tokenizers from a run name.

## Fixed training inputs

The original three simplification `.train` files remain in `data/`. The raw controls, explanation, paraphrase-only and BB26 input files, validation data, exact tokenizer models and base architecture configuration are now included too. `provenance/` contains the source construction manifests. Original source permissions apply to data; the repository's code license does not relicense third-party corpora.

The 100M corpus is split into independent gzip members to stay below GitHub's per-file limit. Restore it with:

```bash
python reproduction/restore_corpus_100m.py
```

The restorer validates every decoded chunk and the complete 615,810,639-byte input with SHA-256 before committing the output. It does not regenerate text. `pairs/*.jsonl.gz` retain the original/rewrite fields needed for the paper's source-level readability and pairing analyses. Negation fields in source records are retained as provenance; the paraphrase-only training file uses only the paraphrase output.

## Train

Use Linux with CUDA and bitsandbytes LAMB; inspect `environment/` and install the dependencies. The launcher's default is to **print** the command, so inspection does not start training.

```bash
python reproduction/train_condition.py --condition fw_simp --masking wwm7tok3 --seed 2 --output output/fw_simp_seed2
# Add --execute to actually train.
python reproduction/train_condition.py --condition submission100m --seed 1 --output output/submission100m
```

The launcher uses the released full `train_mask.py` regular-MLM path, preserving its masking and span implementations. `batch_size` is the total batch, and `grad_acc` splits that batch into microbatches; do not multiply them again when computing tokens per update. Sequence lengths change 64→256 with inverse batch-size scaling. The 10M setting is 384/1280 hidden/intermediate, total batch 256, grad_acc 2; 100M is 768/2560, batch 2048, grad_acc 4 and pure span p=0.5. Final-only saving is selected for compact reproduction; intermediate checkpoints are not necessary to evaluate the final model.

## Evaluate

The evaluation snapshot is in `evaluation/`, based on upstream [BabyLM eval](https://github.com/babylm-org/babylm-eval) commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc` with the saved local pipeline modifications. Benchmark data must be obtained through the official distribution; it is not duplicated here. From `reproduction/evaluation` with its dependencies and `evaluation_data/full_eval` installed:

```bash
bash scripts/eval_zero_shot.sh /absolute/path/to/final-checkpoint mlm
bash scripts/eval_finetuning.sh --model_path /absolute/path/to/final-checkpoint
bash scripts/eval_zero_shot_global_piqa.sh /absolute/path/to/final-checkpoint mlm
```

The historical finetuning script sources `../.env`; create an empty file there if no optional tracking credentials are used. No credentials are included in this release. Hidden-task results depend on the official split and scoring version. Published weights: [10M](https://huggingface.co/go76dof/wwm_curriculum_simplification_40k), [100M](https://huggingface.co/go76dof/spangeo05_simplification).

## Rebuild analyses from saved evidence

```bash
python reproduction/summarize_results.py
python reproduction/bootstrap_seeds.py reproduction/generated_summary.json --condition fw_simp --metric ewok
# Requires textstat; operates on all saved pairs, not a sample:
python reproduction/readability.py reproduction/pairs/fineweb_simplified.jsonl.gz --output fw_readability.json
```

Prompts from paper Appendix B are in `prompts/`; recovered generation/extraction code is in `source_scripts/`. Exact final training files are preserved because stochastic generation and historical preprocessing cannot be certified to regenerate the same bytes. The original temporary readability/bootstrap scripts and their exact dependency/RNG states were not retained. The new analysis scripts explicitly label this limitation. Report-based means use saved rounded scores, so the last decimal may differ from unrounded prediction-based recomputation. No training, model scoring or full historical environment reproduction was rerun as part of preparing this release.

`FILE_MANIFEST.json` supplies SHA-256 and Git blob hashes for the published files. Development caches, unrelated experiments, duplicate models, intermediate checkpoints, private settings, logs and Git garbage are intentionally outside this retention scope.

## Material numerical discrepancy

The pure paraphrase runs have saved Entity means about 36.81 (Token) and 38.29 (WWM), whereas paper Table 5 reports 19.0 and 19.1. The older negation-triplet runs match the latter values. Do not infer that the paper’s selective Entity Tracking conclusion has been reproduced. This release preserves both relevant alternatives and identifies the discrepancy without changing the paper or inventing missing results. The exact cause, including possible evaluation-version differences, is not established by this release.
