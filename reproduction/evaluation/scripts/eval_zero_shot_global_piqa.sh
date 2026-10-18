#!/bin/bash
# GlobalPIQA-only zero-shot evaluation for a single model, covering both the
# main checkpoint and all intermediate (fast-revision) checkpoints. This is a
# combined, GlobalPIQA-dedicated version of eval_zero_shot.sh +
# eval_zero_shot_fast_all_revisions.sh.
#
# GlobalPIQA is not subsampled for the fast evaluation, so the parallel/
# nonparallel data are identical under full_eval and fast_eval. The main
# checkpoint (which feeds the full-eval collation) is scored against full_eval;
# the intermediate checkpoints (which feed the fast-eval collation) are scored
# against fast_eval, mirroring every other fast task. Predictions are
# length-normalized at inference time, so the resulting accuracy is acc_norm.
#
# Usage (run from the strict/ directory):
#   bash scripts/eval_zero_shot_global_piqa.sh MODEL_PATH BACKEND TRACK [EVAL_DIR] [FAST_EVAL_DIR]
#
#   MODEL_PATH     HuggingFace repo id or local path to the model.
#   BACKEND        Evaluation backend (causal, mlm, mntp, enc_dec_mask, enc_dec_prefix).
#   TRACK          strict | strict-small (strict-small stops at the 100M checkpoint).
#   EVAL_DIR       Optional full-eval data dir for the main checkpoint (default evaluation_data/full_eval).
#   FAST_EVAL_DIR  Optional fast-eval data dir for intermediate checkpoints (default evaluation_data/fast_eval).

MODEL_PATH=$1
BACKEND=$2
TRACK=$3
EVAL_DIR=${4:-"evaluation_data/full_eval"}
FAST_EVAL_DIR=${5:-"evaluation_data/fast_eval"}

# Run the two GlobalPIQA tasks for one checkpoint against the given data dir.
# With no revision argument the pipeline defaults to the "main" checkpoint;
# otherwise the given revision name is passed through.
run_global_piqa () {
    local revision=$1
    local data_dir=$2
    local revision_args=""
    if [[ -n "$revision" ]]; then
        revision_args="--revision_name $revision"
    fi

    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task global_piqa_parallel --data_path "${data_dir}/global_piqa_parallel" --save_predictions $revision_args
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task global_piqa_nonparallel --data_path "${data_dir}/global_piqa_nonparallel" --save_predictions $revision_args
}

# Main checkpoint -> full_eval
echo "Evaluating checkpoint main"
run_global_piqa "" "$EVAL_DIR"

# Intermediate checkpoints -> fast_eval: 1M..9M, then 10M..100M
for i in {1..9}; do
    checkpoint="chck_${i}M"
    echo "Evaluating checkpoint ${checkpoint}"
    run_global_piqa $checkpoint "$FAST_EVAL_DIR"
done

for i in {10..100..10}; do
    checkpoint="chck_${i}M"
    echo "Evaluating checkpoint ${checkpoint}"
    run_global_piqa $checkpoint "$FAST_EVAL_DIR"
done

# Full strict track continues past 100M up to 1000M; strict-small stops at 100M.
if [[ "$TRACK" != "strict-small" ]]; then
    for i in {200..1000..100}; do
        checkpoint="chck_${i}M"
        echo "Evaluating checkpoint ${checkpoint}"
        run_global_piqa $checkpoint "$FAST_EVAL_DIR"
    done
fi
