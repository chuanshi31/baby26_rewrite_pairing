# train_mask.py
import argparse
import csv
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoConfig, AutoModelForMaskedLM, DebertaV2Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset, load_from_disk

from preprocessing import tokenize, padding_collate_fn, group_texts, group_texts_with_phrases

from bitsandbytes.optim import LAMB

try:
    import wandb
    wandb_available = True
except Exception:
    wandb_available = False

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="")
parser.add_argument("--valid_data", type=str, default="data/even.dev")
parser.add_argument("--max_seq_len", type=str, default="64", help="Either num or e.g. 0:64,5:128")
parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_acc", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.007)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--cpus", type=int, default=64)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--all_checkpoints", action="store_true", help="Save and evaluate model every 1/10/100M tokens, \
                    per challenge stipulations. Overrides eval_steps and save_steps.")
parser.add_argument("--log_mlm_probs", action="store_true", help="Log MLM probabilities for analysis")
parser.add_argument("--mask_update_steps", type=int, default=200)
parser.add_argument("--first_mask_update", type=int, default=0, help="Do not update mask before this global step")
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--intermediate_size", type=int, default=3072)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mlm_prob", type=float, default=0.15)
parser.add_argument("--mask_replace_prob", type=float, default=0.8)
parser.add_argument("--random_replace_prob", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pretrained", action="store_true", help="Load pretrained model")
parser.add_argument("--eval_only", action="store_true", help="Evaluate only")
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--wandb", action="store_true", help="Report to wandb")
parser.add_argument("--regular_mlm", action="store_true", help="Regular MLM")
parser.add_argument("--custom", type=str, default="", help="Use custom model")
parser.add_argument("--lamb", action="store_true", help="LAMB optimization")
parser.add_argument("--lower", action="store_true", help="Lowercase")
parser.add_argument("--soft", action="store_true", help="Soft mask")
parser.add_argument("--flops", action="store_true", help="Compute FLOPs")
parser.add_argument("--mask_decay", type=float, default=0.0, help="Mask decay. e.g. 0.1 means decay by 0.1 \
                    over the course of training")
parser.add_argument("--span_masking", type=str, default="token",
                    choices=["token", "morph_token", "span_fixed", "span_geo", "span", "wwm", "wwm_token_mix", "wwm_token_plateau", "adaptive_wwm_token", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word", "wwm_mix", "wwm_span_fixed", "wwm_span_geo", "phrase"],
                    help="Masking strategy: token (original), morph_token (inside-word subword biased token masking), span_fixed/span_geo (token spans), wwm (whole word), wwm_token_mix (sample WWM or token per sequence), wwm_token_plateau (WWM until loss plateaus, then token), adaptive_wwm_token (learn global WWM/token ratio from loss EMA), adaptive_wwm_token_word (word-level WWM/token choice from per-word loss EMA), adaptive_wwm_start_inside_word (word-level WWM/start/inside choice), wwm_mix (whole-word based mixed corruption), wwm_span_fixed/wwm_span_geo (whole-word spans)")
parser.add_argument("--span_len", type=int, default=3,
                    help="Fixed span length for span_fixed mode")
parser.add_argument("--span_geo_p", type=float, default=1/3,
                    help="Geometric distribution parameter p for span_geo mode (mean span length = 1/p)")
parser.add_argument("--span_geo_p_late", type=float, default=None,
                    help="If set, span_geo_p switches to this value at --span_geo_p_switch_epoch (within the span_masking curriculum phase). Masking type stays span_geo throughout.")
parser.add_argument("--span_geo_p_switch_epoch", type=int, default=0,
                    help="Epoch at which span_geo_p switches from --span_geo_p to --span_geo_p_late. Only used if --span_geo_p_late is set.")
parser.add_argument("--span_geo_p_ramp_target", type=float, default=None,
                    help="If set, span_geo_p increases stepwise each epoch from --span_geo_p to this value by the final epoch (linear schedule over args.epochs). Masking type stays span_geo throughout. Takes precedence over --span_geo_p_late if both are set.")
parser.add_argument("--span_stats_mode", type=str, default="average",
                    choices=["average", "exact"],
                    help="Stats attribution for span masking: average (split span loss equally) or exact (per-token loss)")
parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to checkpoint directory to resume training from. Resumes at the step encoded in the directory name.")
parser.add_argument("--span_epochs", type=int, default=0,
                    help="Reverse curriculum: use span_masking for the first N epochs, then switch to token masking. "
                         "0 means always use the span_masking setting (no curriculum switch).")
parser.add_argument("--transition_masking", type=str, default="",
                    choices=["", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word", "wwm_token_mix", "wwm_mix"],
                    help="Optional middle stage for reverse curriculum: span_masking -> transition_masking -> post_span_masking.")
parser.add_argument("--transition_epochs", type=int, default=0,
                    help="Number of epochs to use transition_masking after span_epochs before post_span_masking.")
parser.add_argument("--post_span_masking", type=str, default="token",
                    choices=["token", "morph_token"],
                    help="Masking strategy after reverse curriculum switches away from span_masking.")
parser.add_argument("--token_epochs", type=int, default=0,
                    help="Forward curriculum: use token masking for the first N epochs, then switch to span_masking. "
                         "0 means always use the span_masking setting (no curriculum switch).")
parser.add_argument("--morph_inside_weight", type=float, default=5.0,
                    help="Sampling multiplier for inside-word alphabetic subword tokens in morph_token mode.")
parser.add_argument("--morph_start_weight", type=float, default=1.0,
                    help="Sampling multiplier for word-start tokens in morph_token mode.")
parser.add_argument("--wwm_token_mix_wwm_prob", type=float, default=0.80,
                    help="In wwm_token_mix mode, probability that a sequence uses WWM instead of token masking.")
parser.add_argument("--plateau_switch_min_epochs", type=float, default=2.0,
                    help="Minimum WWM epochs before wwm_token_plateau is allowed to switch to token masking.")
parser.add_argument("--plateau_switch_max_epochs", type=float, default=7.0,
                    help="Force wwm_token_plateau to switch to token masking after this many WWM epochs. <=0 disables the force switch.")
parser.add_argument("--plateau_switch_patience", type=int, default=3,
                    help="Number of consecutive low-improvement windows before wwm_token_plateau switches to token masking.")
parser.add_argument("--plateau_switch_threshold", type=float, default=0.01,
                    help="Minimum window loss improvement required to avoid a plateau count.")
parser.add_argument("--plateau_switch_check_steps", type=int, default=0,
                    help="How often to check WWM loss plateau. 0 means use mask_update_steps.")
parser.add_argument("--adaptive_wwm_token_initial_wwm_prob", type=float, default=0.70,
                    help="Initial WWM probability for adaptive_wwm_token mode.")
parser.add_argument("--adaptive_wwm_start_inside_initial_start_prob", type=float, default=0.15,
                    help="Initial start-token probability for adaptive_wwm_start_inside_word mode.")
parser.add_argument("--adaptive_wwm_start_inside_initial_inside_prob", type=float, default=0.15,
                    help="Initial inside-token probability for adaptive_wwm_start_inside_word mode.")
parser.add_argument("--adaptive_wwm_token_warmup_epochs", type=int, default=1,
                    help="Number of initial epochs using the fixed initial WWM probability before adaptive updates.")
parser.add_argument("--adaptive_wwm_token_ema_alpha", type=float, default=0.05,
                    help="EMA update rate for adaptive_wwm_token mode losses.")
parser.add_argument("--adaptive_wwm_token_beta", type=float, default=1.0,
                    help="Weight for positive improvement in adaptive_wwm_token mode scores.")
parser.add_argument("--adaptive_wwm_token_temperature", type=float, default=1.0,
                    help="Softmax temperature for adaptive_wwm_token mode scores.")
parser.add_argument("--adaptive_wwm_token_epsilon_start", type=float, default=0.30,
                    help="Initial random-exploration rate mixed into adaptive WWM/token probabilities.")
parser.add_argument("--adaptive_wwm_token_epsilon_end", type=float, default=0.05,
                    help="Final random-exploration rate mixed into adaptive WWM/token probabilities.")
parser.add_argument("--adaptive_wwm_token_min_wwm_prob", type=float, default=0.40,
                    help="Lower clamp for adaptive WWM probability.")
parser.add_argument("--adaptive_wwm_token_max_wwm_prob", type=float, default=0.80,
                    help="Upper clamp for adaptive WWM probability.")
parser.add_argument("--adaptive_wwm_token_token_improvement_discount", type=float, default=0.50,
                    help="Discount for token improvement because token masking is naturally easier than WWM.")
parser.add_argument("--adaptive_wwm_token_min_count", type=int, default=3,
                    help="Minimum per-mode observations before word-level adaptive WWM/token trusts a word's statistics.")
parser.add_argument("--adaptive_word_stats_update_steps", type=int, default=1,
                    help="Update word-level adaptive WWM/token statistics every N optimizer steps. 1 means every step.")
parser.add_argument("--wwm_mix_whole_prob", type=float, default=0.80,
                    help="In wwm_mix mode, probability of corrupting the whole selected word.")
parser.add_argument("--wwm_mix_inside_prob", type=float, default=0.15,
                    help="In wwm_mix mode, probability of corrupting one inside-word alphabetic token from the selected word.")
parser.add_argument("--wwm_mix_start_prob", type=float, default=0.05,
                    help="In wwm_mix mode, probability of corrupting the word-start token from the selected word.")
parser.add_argument("--wwm_mix_schedule", type=str, default="fixed",
                    choices=["fixed", "linear"],
                    help="Schedule for wwm_mix probabilities. fixed uses wwm_mix_*_prob; linear interpolates start_* to end_* over the wwm_mix phase.")
parser.add_argument("--wwm_mix_start_whole_prob", type=float, default=0.90,
                    help="Linear wwm_mix schedule start probability for whole-word corruption.")
parser.add_argument("--wwm_mix_start_inside_prob", type=float, default=0.08,
                    help="Linear wwm_mix schedule start probability for inside-word corruption.")
parser.add_argument("--wwm_mix_start_start_prob", type=float, default=0.02,
                    help="Linear wwm_mix schedule start probability for word-start corruption.")
parser.add_argument("--wwm_mix_end_whole_prob", type=float, default=0.10,
                    help="Linear wwm_mix schedule end probability for whole-word corruption.")
parser.add_argument("--wwm_mix_end_inside_prob", type=float, default=0.45,
                    help="Linear wwm_mix schedule end probability for inside-word corruption.")
parser.add_argument("--wwm_mix_end_start_prob", type=float, default=0.45,
                    help="Linear wwm_mix schedule end probability for word-start corruption.")
parser.add_argument("--curriculum_lr_reset", action="store_true",
                    help="Reset LR to 0.7x with new cosine schedule when curriculum switches masking strategy.")
parser.add_argument("--sbo", action="store_true", help="Add Span Boundary Objective (SpanBERT-style) loss when using span masking")
parser.add_argument("--wwm_pos", action="store_true",
                    help="Add a lightweight word-internal position auxiliary loss for WWM-style masking.")
parser.add_argument("--wwm_pos_weight", type=float, default=0.1,
                    help="Weight for the word-internal position auxiliary loss.")
parser.add_argument("--wwm_pos_modes", type=str, default="wwm,wwm_token_mix,wwm_mix,wwm_span_fixed,wwm_span_geo",
                    help="Comma-separated active masking modes for the word-internal position auxiliary loss.")
parser.add_argument("--wwm_pos_scheme", type=str, default="binary",
                    choices=["binary", "single_start_middle_end"],
                    help="Word-internal position scheme for --wwm_pos. binary uses start/inside; single_start_middle_end uses single/start/middle/end.")
parser.add_argument("--freeze_lower_layers", type=int, default=0,
                    help="Freeze encoder layers 0..N-1 (and embeddings) at training start. "
                         "0 means no freezing. E.g. 6 freezes layer 0-5, trains layer 6-11 + mlm_head + relative_position.")
parser.add_argument("--freeze_from_epoch", type=int, default=-1,
                    help="Epoch at which to apply freeze_lower_layers / freeze_layers. "
                         "-1 means freeze at training start (default behavior). "
                         "E.g. 7 means freeze only from epoch 7 onwards (useful for curriculum freeze).")
parser.add_argument("--phrase_data", type=str, default="",
                    help="Path to pre-parsed HuggingFace dataset with phrase_ids column "
                         "(produced by scripts/parse_phrases.py). Required when --span_masking phrase.")
parser.add_argument("--freeze_layers", type=str, default="",
                    help="Freeze specific encoder layer indices (comma-separated) + embeddings. "
                         "E.g. '0,1,2,3,4,7,8,9' freezes those layers regardless of contiguity. "
                         "Takes precedence over --freeze_lower_layers if non-empty.")

def evaluate(model, tokenizer, dataloader, args):
    AMP_DTYPE = get_amp_dtype()
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if len(batch["input_ids"]) == 0:
                continue # NOTE: not sure why this happens in 100M case...

            masked_batch = mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15,
                                      mask_replace_prob=0.8, random_replace_prob=0.1,
                                      span_masking=args._active_span_masking, span_len=args.span_len,
                                      span_geo_p=args.span_geo_p, span_stats_mode=args.span_stats_mode,
                                      morph_inside_weight=args.morph_inside_weight,
                                      morph_start_weight=args.morph_start_weight,
                                      wwm_token_mix_wwm_prob=args.wwm_token_mix_wwm_prob,
                                      adaptive_wwm_token_wwm_prob=getattr(args, "_active_adaptive_wwm_prob", args.adaptive_wwm_token_initial_wwm_prob),
                                      adaptive_word_stats=getattr(args, "_adaptive_word_stats_cpu", None),
                                      adaptive_epoch=getattr(args, "_eval_epoch", args.epochs),
                                      adaptive_args=args,
                                      wwm_mix_whole_prob=getattr(args, "_active_wwm_mix_whole_prob", args.wwm_mix_whole_prob),
                                      wwm_mix_inside_prob=getattr(args, "_active_wwm_mix_inside_prob", args.wwm_mix_inside_prob),
                                      wwm_mix_start_prob=getattr(args, "_active_wwm_mix_start_prob", args.wwm_mix_start_prob))
            # split batch into grad_acc chunks
            batches = split_batch(masked_batch, args)
            for minibatch in batches:
                with torch.autocast(dtype=AMP_DTYPE, device_type="cuda:0"):
                    model_inputs = {k: v for k, v in move_dict_to_cuda(minibatch).items()
                                    if k not in ("original_ids", "span_ids", "span_boundaries", "mask_mode_ids", "mask_mode_pos_ids", "word_key_ids", "adaptive_stats", "phrase_ids")}
                    outputs = model(**model_inputs)
            
                avg_loss += outputs.loss.item()
                logits = outputs.logits
                preds = logits.argmax(dim=-1)

                labels = minibatch["labels"].to(device=logits.device, dtype=logits.dtype)
                label_mask = labels != -100
                correct += (preds[label_mask] == labels[label_mask]).sum().item()
                total += preds[label_mask].numel()

    model.train()
    return {'acc': 100 * correct / total if total > 0 else 0.0,
            'loss': avg_loss / (len(dataloader) * args.grad_acc)}


def regroup_texts(args, max_seq_len):
    cur_max_seq_len = args.cur_max_seq_len
    print("CUR MAX SEQ LEN:", cur_max_seq_len)

    def map_with_fallback(dataset, fn, *, batched, fn_kwargs, num_proc, desc):
        try:
            return dataset.map(
                fn,
                batched=batched,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                desc=desc,
            )
        except Exception as exc:
            if num_proc == 1:
                raise
            print(
                f"[SeqLen] {desc} with num_proc={num_proc} failed: {repr(exc)}. "
                "Retrying with num_proc=1.",
                flush=True,
            )
            return dataset.map(
                fn,
                batched=batched,
                fn_kwargs=fn_kwargs,
                num_proc=1,
                desc=f"{desc} (single process retry)",
            )

    use_phrase = getattr(args, "_use_phrase_grouping", False)
    if use_phrase:
        # Train split has phrase_ids; validation split does not — use separate group fns
        grouped_train = map_with_fallback(args.dataset['train'], group_texts_with_phrases,
            batched=True,
            fn_kwargs={'max_len': max_seq_len},
            num_proc=1,
            desc="Grouping train",
        )
        grouped_val = map_with_fallback(args.dataset['validation'], group_texts,
            batched=True,
            fn_kwargs={'max_len': max_seq_len},
            num_proc=args.cpus,
            desc="Grouping validation",
        )
        from datasets import DatasetDict
        grouped_dataset = DatasetDict({"train": grouped_train, "validation": grouped_val})
    else:
        grouped_dataset = map_with_fallback(args.dataset, group_texts,
            batched=True,
            fn_kwargs={'max_len': max_seq_len},
            num_proc=args.cpus,
            desc="Grouping",
        )
    change_ratio = max_seq_len / cur_max_seq_len
    args.batch_size = max(1, int(args.batch_size / change_ratio))
    
    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn
        )

    args.cur_max_seq_len = max_seq_len

    return train_dataloader, eval_dataloader

def _apply_mask_tokens(enc, enc_mask_mask, tokenizer, mask_replace_prob, random_replace_prob):
    """Replace selected positions with [MASK] or random token according to BERT 80/10/10 rule."""
    rand = torch.rand(len(enc))
    to_mask = (rand < mask_replace_prob) & enc_mask_mask
    to_replace = (rand >= mask_replace_prob) & (rand < mask_replace_prob + random_replace_prob) & enc_mask_mask
    randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
    enc[to_mask] = tokenizer.mask_token_id
    enc[to_replace] = randoms[to_replace]
    return enc


def _apply_mask_groups(enc, selected, group_ids, tokenizer, mask_replace_prob, random_replace_prob):
    """
    Apply the BERT 80/10/10 replacement rule at group level.
    All tokens with the same non-negative group_id receive the same replacement type.
    """
    if not selected.any():
        return enc

    max_group = int(group_ids[selected].max().item())
    if max_group < 0:
        return enc

    group_rand = torch.rand(max_group + 1)
    group_to_mask = group_rand < mask_replace_prob
    group_to_replace = (group_rand >= mask_replace_prob) & (group_rand < mask_replace_prob + random_replace_prob)

    active_group_ids = group_ids.clamp(min=0)
    to_mask = selected & group_to_mask[active_group_ids]
    to_replace = selected & group_to_replace[active_group_ids]

    randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
    enc[to_mask] = tokenizer.mask_token_id
    enc[to_replace] = randoms[to_replace]
    return enc


def _morph_token_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob,
                              tokenizer, inside_weight, start_weight):
    """
    Token-level masking biased toward inside-word alphabetic subword tokens.
    This is a light-weight morphology proxy for SentencePiece/BPE tokenizers:
    tokens with no leading ▁ and alphabetic surface form receive inside_weight.
    """
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc]
    is_alpha = tokenizer.vocab_is_alpha[enc]
    maskable = ~not_maskable
    inside_word = maskable & ~is_word_start & is_alpha
    word_start = maskable & is_word_start

    morph_weights = torch.ones(seq_len, dtype=enc_mask_weights.dtype)
    morph_weights[word_start] = start_weight
    morph_weights[inside_word] = inside_weight
    morph_weights[~maskable] = 0.0

    biased_weights = enc_mask_weights.float() * morph_weights.float()
    weight_sum = biased_weights.sum()
    if weight_sum <= 0:
        return torch.zeros(seq_len, dtype=torch.bool)

    probs = mlm_prob * seq_len * biased_weights / weight_sum
    probs = probs.clamp(max=1.0)
    return torch.rand(seq_len) < probs


def _wwm_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob,
                      tokenizer, use_weights):
    """
    Whole Word Masking（全向量化实现）。
    以完整词为单位采样，无任何 Python 循环。
    """
    # 直接用预计算的词表级 tensor 索引，无 Python 字符串操作
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc]
    # 只在可 mask 的位置上判断词首，不可 mask 的位置排除在外
    # 第一个可 mask 的 token 强制设为词首
    maskable_pos = (~not_maskable).nonzero(as_tuple=True)[0]
    if len(maskable_pos) == 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)
    is_word_start[maskable_pos[0]] = True

    # 只在可 mask 位置上做 cumsum，不可 mask 位置的 is_word_start 不参与计数
    is_word_start_maskable = is_word_start & ~not_maskable
    # 累加得到每个位置的词编号（0-based），不可 mask 位置设为 -1
    word_ids_raw = is_word_start_maskable.long().cumsum(0) - 1
    word_ids = torch.where(not_maskable, torch.full_like(word_ids_raw, -1), word_ids_raw)

    num_words = word_ids.max().item() + 1
    if num_words <= 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]

    # 每个词的 token 数
    word_token_counts = torch.zeros(num_words, dtype=torch.long)
    word_token_counts.scatter_add_(0, maskable_wids, torch.ones(maskable.sum(), dtype=torch.long))

    # 每个词的权重
    if use_weights:
        word_weights = torch.zeros(num_words)
        word_weights.scatter_add_(0, maskable_wids, enc_mask_weights[maskable].float())
        word_weights = word_weights / word_token_counts.float().clamp(min=1)
        if word_weights.sum() == 0:
            word_weights = torch.ones(num_words)
    else:
        word_weights = torch.ones(num_words)

    # 一次采样所有词的排列，按 token 数累加截断
    word_order = torch.multinomial(word_weights, num_words, replacement=False)
    cumsum = word_token_counts[word_order].cumsum(0)
    target = int(mlm_prob * seq_len)
    n_select = min((cumsum <= target).sum().item() + 1, num_words)

    # 用 scatter 构建 word → span_idx 映射
    span_map = torch.full((num_words,), -1, dtype=torch.long)
    selected_wids = word_order[:n_select]
    span_map[selected_wids] = torch.arange(n_select, dtype=torch.long)

    # 向量化映射回每个位置
    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    span_ids[maskable] = span_map[maskable_wids]
    selected = span_ids >= 0

    return selected, span_ids


def _phrase_mask_indices(seq_len, pad_token_id, enc, phrase_ids_enc, mlm_prob, tokenizer):
    """
    Phrase masking: treat each constituent phrase as a unit (like WWM treats words).
    phrase_ids_enc: LongTensor of shape (seq_len,), -1 for tokens not in any phrase
                    or special/pad tokens.
    Falls back to token masking for positions with phrase_id == -1.

    Returns (selected: BoolTensor, span_ids: LongTensor) same contract as _wwm_mask_indices.
    """
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    # Tokens that belong to a phrase
    has_phrase = (phrase_ids_enc >= 0) & ~not_maskable
    # Tokens with no phrase assignment — treated as individual units (fallback)
    no_phrase   = (~has_phrase) & ~not_maskable

    # ── Build unit list ───────────────────────────────────────────────────────
    # Each phrase group is one unit; each unphrase token is its own unit.
    # unit_ids[i] = unit index for position i, -1 if not maskable
    num_phrases = phrase_ids_enc[has_phrase].max().item() + 1 if has_phrase.any() else 0

    # Phrase tokens: unit_id = phrase_id (0-based, already re-indexed by group_texts_with_phrases)
    # No-phrase tokens: unit_id = num_phrases + running_index
    unit_ids = torch.full((seq_len,), -1, dtype=torch.long)
    if has_phrase.any():
        unit_ids[has_phrase] = phrase_ids_enc[has_phrase]

    no_phrase_positions = no_phrase.nonzero(as_tuple=True)[0]
    num_solo = len(no_phrase_positions)
    if num_solo > 0:
        unit_ids[no_phrase_positions] = torch.arange(num_phrases, num_phrases + num_solo, dtype=torch.long)

    num_units = num_phrases + num_solo
    if num_units == 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)

    maskable     = unit_ids >= 0
    maskable_uids = unit_ids[maskable]

    # Token count per unit
    unit_token_counts = torch.zeros(num_units, dtype=torch.long)
    unit_token_counts.scatter_add_(0, maskable_uids, torch.ones(maskable.sum(), dtype=torch.long))

    # Sample units in random order, greedily include until target reached
    unit_order = torch.randperm(num_units)
    cumsum     = unit_token_counts[unit_order].cumsum(0)
    target     = int(mlm_prob * seq_len)
    n_select   = min((cumsum <= target).sum().item() + 1, num_units)

    span_map = torch.full((num_units,), -1, dtype=torch.long)
    selected_uids = unit_order[:n_select]
    span_map[selected_uids] = torch.arange(n_select, dtype=torch.long)

    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    span_ids[maskable] = span_map[maskable_uids]
    selected = span_ids >= 0

    return selected, span_ids


def _wwm_mix_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob,
                          tokenizer, use_weights, whole_prob, inside_prob, start_prob):
    """
    Whole-word based mixed masking.

    First sample words as in WWM. For each selected word, choose one corruption type:
      - whole: mask every token in the word
      - inside: mask one alphabetic inside-word subword token if available
      - start: mask only the word-start token

    If the requested partial corruption is impossible for a word, fall back to whole-word
    corruption so the effective MLM budget stays close to regular WWM.
    """
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc]
    maskable_pos = (~not_maskable).nonzero(as_tuple=True)[0]
    if len(maskable_pos) == 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)
    is_word_start[maskable_pos[0]] = True

    is_word_start_maskable = is_word_start & ~not_maskable
    word_ids_raw = is_word_start_maskable.long().cumsum(0) - 1
    word_ids = torch.where(not_maskable, torch.full_like(word_ids_raw, -1), word_ids_raw)

    num_words = word_ids.max().item() + 1
    if num_words <= 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]

    word_token_counts = torch.zeros(num_words, dtype=torch.long)
    word_token_counts.scatter_add_(0, maskable_wids, torch.ones(maskable.sum(), dtype=torch.long))

    if use_weights:
        word_weights = torch.zeros(num_words)
        word_weights.scatter_add_(0, maskable_wids, enc_mask_weights[maskable].float())
        word_weights = word_weights / word_token_counts.float().clamp(min=1)
        if word_weights.sum() == 0:
            word_weights = torch.ones(num_words)
    else:
        word_weights = torch.ones(num_words)

    word_order = torch.multinomial(word_weights, num_words, replacement=False)
    cumsum = word_token_counts[word_order].cumsum(0)
    target = int(mlm_prob * seq_len)
    n_select = min((cumsum <= target).sum().item() + 1, num_words)
    selected_wids = word_order[:n_select]

    probs = torch.tensor([whole_prob, inside_prob, start_prob], dtype=torch.float)
    if probs.sum() <= 0:
        probs = torch.tensor([1.0, 0.0, 0.0])
    probs = probs / probs.sum()

    selected = torch.zeros(seq_len, dtype=torch.bool)
    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    is_alpha = tokenizer.vocab_is_alpha[enc]

    for span_idx, wid in enumerate(selected_wids.tolist()):
        positions = (word_ids == wid).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            continue

        mode = torch.multinomial(probs, 1).item()
        word_start_positions = positions[is_word_start[positions]]
        if len(word_start_positions) == 0:
            word_start_positions = positions[:1]
        inside_positions = positions[(~is_word_start[positions]) & is_alpha[positions]]

        if mode == 1 and len(inside_positions) > 0:
            chosen = inside_positions[torch.randint(len(inside_positions), (1,)).item()].view(1)
        elif mode == 2 and len(word_start_positions) > 0:
            chosen = word_start_positions[:1]
        else:
            chosen = positions

        selected[chosen] = True
        span_ids[chosen] = span_idx

    return selected, span_ids


def _wwm_span_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob,
                           tokenizer, span_mode, span_len, span_geo_p, use_weights):
    """
    Whole-word span masking.
    Sample contiguous spans in word space, then mask every subword covered by those words.

    span_mode: "fixed" or "geo"; length is measured in complete words, not tokens.
    """
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc]

    maskable_pos = (~not_maskable).nonzero(as_tuple=True)[0]
    if len(maskable_pos) == 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)
    is_word_start[maskable_pos[0]] = True

    is_word_start_maskable = is_word_start & ~not_maskable
    word_ids_raw = is_word_start_maskable.long().cumsum(0) - 1
    word_ids = torch.where(not_maskable, torch.full_like(word_ids_raw, -1), word_ids_raw)

    num_words = word_ids.max().item() + 1
    if num_words <= 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.full((seq_len,), -1, dtype=torch.long)

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]

    word_token_counts = torch.zeros(num_words, dtype=torch.long)
    word_token_counts.scatter_add_(0, maskable_wids, torch.ones(maskable.sum(), dtype=torch.long))

    if use_weights:
        word_weights = torch.zeros(num_words)
        word_weights.scatter_add_(0, maskable_wids, enc_mask_weights[maskable].float())
        word_weights = word_weights / word_token_counts.float().clamp(min=1)
        if word_weights.sum() == 0:
            word_weights = torch.ones(num_words)
    else:
        word_weights = torch.ones(num_words)

    target = int(mlm_prob * seq_len)
    selected_words = torch.zeros(num_words, dtype=torch.bool)
    word_span_ids = torch.full((num_words,), -1, dtype=torch.long)
    selected_count = 0
    span_idx = 0

    for _ in range(num_words * 2):
        if selected_count >= target:
            break

        if span_mode == "fixed":
            Lw = span_len
        else:
            Lw = min(int(np.random.geometric(span_geo_p)), num_words)

        max_start = num_words - Lw
        if max_start < 0:
            continue

        starts = torch.arange(max_start + 1)
        selected_prefix = torch.cat([
            torch.zeros(1, dtype=torch.long),
            selected_words.long().cumsum(0)
        ])
        available = (selected_prefix[Lw:] - selected_prefix[:-Lw]) == 0

        if not available.any():
            break

        if use_weights:
            weight_prefix = torch.cat([
                torch.zeros(1, dtype=word_weights.dtype),
                word_weights.cumsum(0)
            ])
            window_w = (weight_prefix[Lw:] - weight_prefix[:-Lw]) / Lw
            window_w[~available] = 0.0
            if window_w.sum() == 0:
                break
            start = torch.multinomial(window_w, 1).item()
        else:
            valid_starts = starts[available]
            start = valid_starts[torch.randint(len(valid_starts), (1,)).item()].item()

        selected_words[start:start + Lw] = True
        word_span_ids[start:start + Lw] = span_idx
        span_idx += 1
        selected_count = word_token_counts[selected_words].sum().item()

    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    span_ids[maskable] = word_span_ids[maskable_wids]
    selected = span_ids >= 0

    return selected, span_ids


def _span_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob,
                       span_mode, span_len, span_geo_p, use_weights):
    """
    Return a boolean mask of positions selected for masking using span sampling,
    and a span_ids tensor where each masked position holds its span index (0-based).
    Unmasked positions have span_ids = -1.

    span_mode: "fixed" or "geo"
    use_weights: if True, use enc_mask_weights to bias span start selection (AMLM mode);
                 if False, sample start uniformly (classic MLM mode)
    """
    target = int(mlm_prob * seq_len)
    selected = torch.zeros(seq_len, dtype=torch.bool)
    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    non_pad = (enc != pad_token_id)
    non_pad_cumsum = torch.cat([torch.zeros(1, dtype=torch.long), non_pad.long().cumsum(0)])
    span_idx = 0
    selected_count = 0

    for _ in range(seq_len * 2):
        if selected_count >= target:
            break

        if span_mode == "fixed":
            L = span_len
        else:
            L = min(int(np.random.geometric(span_geo_p)), seq_len)

        max_start = seq_len - L
        if max_start < 0:
            continue

        s_idx = torch.arange(max_start + 1)
        has_non_pad = (non_pad_cumsum[s_idx + L] - non_pad_cumsum[s_idx]) > 0

        if use_weights:
            weight_prefix = torch.cat([
                torch.zeros(1, dtype=enc_mask_weights.dtype),
                enc_mask_weights.cumsum(0)
            ])
            window_w = (weight_prefix[L:] - weight_prefix[:-L]) / L
            window_w[~has_non_pad] = 0.0
            if window_w.sum() == 0:
                continue
            start = torch.multinomial(window_w, 1).item()
        else:
            valid_starts = s_idx[has_non_pad]
            if len(valid_starts) == 0:
                continue
            start = valid_starts[torch.randint(len(valid_starts), (1,)).item()].item()

        selected[start:start + L] = True
        span_ids[start:start + L] = span_idx
        span_idx += 1
        selected_count = selected.sum().item()

    # Never mask pad tokens
    selected = selected & non_pad
    span_ids[~selected] = -1
    return selected, span_ids


def _word_ids_and_key_ids(seq_len, pad_token_id, enc, tokenizer):
    """Return word ids and word-key ids. The key is the word-start token id."""
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc]
    maskable_pos = (~not_maskable).nonzero(as_tuple=True)[0]
    if len(maskable_pos) == 0:
        return (
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )

    is_word_start[maskable_pos[0]] = True
    is_word_start_maskable = is_word_start & ~not_maskable
    word_ids_raw = is_word_start_maskable.long().cumsum(0) - 1
    word_ids = torch.where(not_maskable, torch.full_like(word_ids_raw, -1), word_ids_raw)

    num_words = word_ids.max().item() + 1
    if num_words <= 0:
        return (
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]
    word_token_counts = torch.zeros(num_words, dtype=torch.long)
    word_token_counts.scatter_add_(0, maskable_wids, torch.ones(maskable.sum(), dtype=torch.long))

    word_start_positions = is_word_start_maskable.nonzero(as_tuple=True)[0]
    word_key_ids = enc[word_start_positions[:num_words]].long()
    word_key_ids_per_pos = torch.full((seq_len,), -1, dtype=torch.long)
    word_key_ids_per_pos[maskable] = word_key_ids[maskable_wids]

    return word_ids, word_key_ids_per_pos, word_key_ids, word_token_counts


def _adaptive_word_wwm_probability(word_key, adaptive_word_stats, args, epoch):
    initial = args.adaptive_wwm_token_initial_wwm_prob
    if adaptive_word_stats is None or epoch < args.adaptive_wwm_token_warmup_epochs:
        return min(max(initial, args.adaptive_wwm_token_min_wwm_prob), args.adaptive_wwm_token_max_wwm_prob)

    wid = int(word_key)
    wwm_count = adaptive_word_stats["wwm_count"][wid].item()
    token_count = adaptive_word_stats["token_count"][wid].item()
    if wwm_count < args.adaptive_wwm_token_min_count or token_count < args.adaptive_wwm_token_min_count:
        return min(max(initial, args.adaptive_wwm_token_min_wwm_prob), args.adaptive_wwm_token_max_wwm_prob)

    max_loss = max(adaptive_word_stats["max_loss"], 1e-8)
    wwm_score = (
        adaptive_word_stats["wwm_loss_ema"][wid].item() / max_loss
        + args.adaptive_wwm_token_beta
        * max(adaptive_word_stats["wwm_improvement"][wid].item(), 0.0)
        / max_loss
    )
    token_score = (
        adaptive_word_stats["token_loss_ema"][wid].item() / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * max(adaptive_word_stats["token_improvement"][wid].item(), 0.0)
        / max_loss
    )

    scores = torch.tensor([wwm_score, token_score], dtype=torch.float32)
    probs = torch.softmax(scores / max(args.adaptive_wwm_token_temperature, 1e-6), dim=0)
    adaptive_p = probs[0].item()
    epsilon = _adaptive_epsilon(args, epoch)
    mixed_p = (1.0 - epsilon) * adaptive_p + epsilon * 0.5
    return min(max(mixed_p, args.adaptive_wwm_token_min_wwm_prob), args.adaptive_wwm_token_max_wwm_prob)


def adaptive_word_wwm_probabilities(adaptive_word_stats, args, epoch):
    """Return the current per-word-key WWM probability table."""
    initial = min(
        max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
        args.adaptive_wwm_token_max_wwm_prob,
    )
    if adaptive_word_stats is None:
        return torch.empty(0, dtype=torch.float32)

    vocab_size = adaptive_word_stats["wwm_count"].shape[0]
    probs = torch.full((vocab_size,), initial, dtype=torch.float32)
    if epoch < args.adaptive_wwm_token_warmup_epochs:
        return probs

    valid = (
        (adaptive_word_stats["wwm_count"] >= args.adaptive_wwm_token_min_count)
        & (adaptive_word_stats["token_count"] >= args.adaptive_wwm_token_min_count)
    )
    if not valid.any():
        return probs

    max_loss = max(adaptive_word_stats["max_loss"], 1e-8)
    wwm_score = (
        adaptive_word_stats["wwm_loss_ema"][valid] / max_loss
        + args.adaptive_wwm_token_beta
        * adaptive_word_stats["wwm_improvement"][valid].clamp(min=0.0)
        / max_loss
    )
    token_score = (
        adaptive_word_stats["token_loss_ema"][valid] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["token_improvement"][valid].clamp(min=0.0)
        / max_loss
    )
    scores = torch.stack([wwm_score, token_score], dim=1)
    adaptive_p = torch.softmax(scores / max(args.adaptive_wwm_token_temperature, 1e-6), dim=1)[:, 0]
    epsilon = _adaptive_epsilon(args, epoch)
    mixed_p = (1.0 - epsilon) * adaptive_p + epsilon * 0.5
    probs[valid] = mixed_p.clamp(
        min=args.adaptive_wwm_token_min_wwm_prob,
        max=args.adaptive_wwm_token_max_wwm_prob,
    )
    return probs


def adaptive_word_wwm_start_inside_probabilities(adaptive_word_stats, args, epoch):
    """Return per-word-key probabilities for WWM/start/inside modes."""
    if adaptive_word_stats is None:
        return tuple(torch.empty(0, dtype=torch.float32) for _ in range(3))

    vocab_size = adaptive_word_stats["wwm_count"].shape[0]
    initial_wwm = min(
        max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
        args.adaptive_wwm_token_max_wwm_prob,
    )
    initial_start = max(args.adaptive_wwm_start_inside_initial_start_prob, 0.0)
    initial_inside = max(args.adaptive_wwm_start_inside_initial_inside_prob, 0.0)
    total = max(initial_wwm + initial_start + initial_inside, 1e-8)
    initial_wwm, initial_start, initial_inside = (
        initial_wwm / total,
        initial_start / total,
        initial_inside / total,
    )

    p_wwm = torch.full((vocab_size,), initial_wwm, dtype=torch.float32)
    p_start = torch.full((vocab_size,), initial_start, dtype=torch.float32)
    p_inside = torch.full((vocab_size,), initial_inside, dtype=torch.float32)
    if epoch < args.adaptive_wwm_token_warmup_epochs:
        return p_wwm, p_start, p_inside

    valid = (
        (adaptive_word_stats["wwm_count"] >= args.adaptive_wwm_token_min_count)
        & (adaptive_word_stats["start_count"] >= args.adaptive_wwm_token_min_count)
        & (adaptive_word_stats["inside_count"] >= args.adaptive_wwm_token_min_count)
    )
    if not valid.any():
        return p_wwm, p_start, p_inside

    max_loss = max(adaptive_word_stats["max_loss"], 1e-8)
    wwm_score = (
        adaptive_word_stats["wwm_loss_ema"][valid] / max_loss
        + args.adaptive_wwm_token_beta
        * adaptive_word_stats["wwm_improvement"][valid].clamp(min=0.0)
        / max_loss
    )
    start_score = (
        adaptive_word_stats["start_loss_ema"][valid] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["start_improvement"][valid].clamp(min=0.0)
        / max_loss
    )
    inside_score = (
        adaptive_word_stats["inside_loss_ema"][valid] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["inside_improvement"][valid].clamp(min=0.0)
        / max_loss
    )
    scores = torch.stack([wwm_score, start_score, inside_score], dim=1)
    probs = torch.softmax(scores / max(args.adaptive_wwm_token_temperature, 1e-6), dim=1)
    epsilon = _adaptive_epsilon(args, epoch)
    probs = (1.0 - epsilon) * probs + epsilon / 3.0

    clipped_wwm = probs[:, 0].clamp(
        min=args.adaptive_wwm_token_min_wwm_prob,
        max=args.adaptive_wwm_token_max_wwm_prob,
    )
    token_total = (1.0 - clipped_wwm).clamp(min=0.0)
    token_raw = probs[:, 1:].sum(dim=1).clamp(min=1e-8)
    p_wwm[valid] = clipped_wwm
    p_start[valid] = token_total * probs[:, 1] / token_raw
    p_inside[valid] = token_total * probs[:, 2] / token_raw
    return p_wwm, p_start, p_inside


def _adaptive_word_wwm_start_inside_probability(word_key, adaptive_word_stats, args, epoch):
    """Return WWM/start/inside probabilities for one word key."""
    initial_wwm = min(
        max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
        args.adaptive_wwm_token_max_wwm_prob,
    )
    initial_start = max(args.adaptive_wwm_start_inside_initial_start_prob, 0.0)
    initial_inside = max(args.adaptive_wwm_start_inside_initial_inside_prob, 0.0)
    total = max(initial_wwm + initial_start + initial_inside, 1e-8)
    default_probs = torch.tensor(
        [initial_wwm / total, initial_start / total, initial_inside / total],
        dtype=torch.float32,
    )
    if (
        adaptive_word_stats is None
        or word_key < 0
        or word_key >= adaptive_word_stats["wwm_count"].shape[0]
        or epoch < args.adaptive_wwm_token_warmup_epochs
    ):
        return default_probs

    has_stats = (
        adaptive_word_stats["wwm_count"][word_key] >= args.adaptive_wwm_token_min_count
        and adaptive_word_stats["start_count"][word_key] >= args.adaptive_wwm_token_min_count
        and adaptive_word_stats["inside_count"][word_key] >= args.adaptive_wwm_token_min_count
    )
    if not has_stats:
        return default_probs

    max_loss = max(adaptive_word_stats["max_loss"], 1e-8)
    wwm_score = (
        adaptive_word_stats["wwm_loss_ema"][word_key] / max_loss
        + args.adaptive_wwm_token_beta
        * adaptive_word_stats["wwm_improvement"][word_key].clamp(min=0.0)
        / max_loss
    )
    start_score = (
        adaptive_word_stats["start_loss_ema"][word_key] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["start_improvement"][word_key].clamp(min=0.0)
        / max_loss
    )
    inside_score = (
        adaptive_word_stats["inside_loss_ema"][word_key] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["inside_improvement"][word_key].clamp(min=0.0)
        / max_loss
    )
    scores = torch.stack([wwm_score, start_score, inside_score])
    probs = torch.softmax(scores / max(args.adaptive_wwm_token_temperature, 1e-6), dim=0)
    epsilon = _adaptive_epsilon(args, epoch)
    probs = (1.0 - epsilon) * probs + epsilon / 3.0

    p_wwm = probs[0].clamp(
        min=args.adaptive_wwm_token_min_wwm_prob,
        max=args.adaptive_wwm_token_max_wwm_prob,
    )
    token_total = (1.0 - p_wwm).clamp(min=0.0)
    token_raw = probs[1:].sum().clamp(min=1e-8)
    return torch.stack([p_wwm, token_total * probs[1] / token_raw, token_total * probs[2] / token_raw])


def _adaptive_word_wwm_start_inside_probs_for_keys(word_keys, adaptive_word_stats, args, epoch):
    """Vectorized WWM/start/inside probabilities for the word keys in one sequence."""
    initial_wwm = min(
        max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
        args.adaptive_wwm_token_max_wwm_prob,
    )
    initial_start = max(args.adaptive_wwm_start_inside_initial_start_prob, 0.0)
    initial_inside = max(args.adaptive_wwm_start_inside_initial_inside_prob, 0.0)
    total = max(initial_wwm + initial_start + initial_inside, 1e-8)
    default = torch.tensor(
        [initial_wwm / total, initial_start / total, initial_inside / total],
        dtype=torch.float32,
    )
    probs_out = default.repeat(len(word_keys), 1)
    if (
        adaptive_word_stats is None
        or len(word_keys) == 0
        or epoch < args.adaptive_wwm_token_warmup_epochs
    ):
        return probs_out

    word_keys = word_keys.long()
    in_vocab = (word_keys >= 0) & (word_keys < adaptive_word_stats["wwm_count"].shape[0])
    if not in_vocab.any():
        return probs_out

    valid_keys = word_keys[in_vocab]
    enough_stats = (
        (adaptive_word_stats["wwm_count"][valid_keys] >= args.adaptive_wwm_token_min_count)
        & (adaptive_word_stats["start_count"][valid_keys] >= args.adaptive_wwm_token_min_count)
        & (adaptive_word_stats["inside_count"][valid_keys] >= args.adaptive_wwm_token_min_count)
    )
    if not enough_stats.any():
        return probs_out

    valid_rows = in_vocab.nonzero(as_tuple=True)[0][enough_stats]
    valid_keys = valid_keys[enough_stats]
    max_loss = max(adaptive_word_stats["max_loss"], 1e-8)
    wwm_score = (
        adaptive_word_stats["wwm_loss_ema"][valid_keys] / max_loss
        + args.adaptive_wwm_token_beta
        * adaptive_word_stats["wwm_improvement"][valid_keys].clamp(min=0.0)
        / max_loss
    )
    start_score = (
        adaptive_word_stats["start_loss_ema"][valid_keys] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["start_improvement"][valid_keys].clamp(min=0.0)
        / max_loss
    )
    inside_score = (
        adaptive_word_stats["inside_loss_ema"][valid_keys] / max_loss
        + args.adaptive_wwm_token_beta
        * args.adaptive_wwm_token_token_improvement_discount
        * adaptive_word_stats["inside_improvement"][valid_keys].clamp(min=0.0)
        / max_loss
    )
    scores = torch.stack([wwm_score, start_score, inside_score], dim=1)
    probs = torch.softmax(scores / max(args.adaptive_wwm_token_temperature, 1e-6), dim=1)
    epsilon = _adaptive_epsilon(args, epoch)
    probs = (1.0 - epsilon) * probs + epsilon / 3.0

    clipped_wwm = probs[:, 0].clamp(
        min=args.adaptive_wwm_token_min_wwm_prob,
        max=args.adaptive_wwm_token_max_wwm_prob,
    )
    token_total = (1.0 - clipped_wwm).clamp(min=0.0)
    token_raw = probs[:, 1:].sum(dim=1).clamp(min=1e-8)
    probs_out[valid_rows, 0] = clipped_wwm
    probs_out[valid_rows, 1] = token_total * probs[:, 1] / token_raw
    probs_out[valid_rows, 2] = token_total * probs[:, 2] / token_raw
    return probs_out


def _adaptive_wwm_token_word_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights,
                                          mlm_prob, tokenizer, use_weights,
                                          adaptive_word_stats, args, epoch):
    """
    Word-level adaptive WWM/token masking.

    Sample candidate words, then decide per word whether to mask the whole word
    or one token from that word using the word's own WWM/token loss statistics.
    """
    word_ids, word_key_ids_per_pos, word_key_ids, word_token_counts = _word_ids_and_key_ids(
        seq_len, pad_token_id, enc, tokenizer
    )
    num_words = len(word_key_ids)
    if num_words <= 0:
        return (
            torch.zeros(seq_len, dtype=torch.bool),
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.full((seq_len,), -1, dtype=torch.long),
            word_key_ids_per_pos,
            torch.zeros(4, dtype=torch.float32),
        )

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]

    if use_weights:
        word_weights = torch.zeros(num_words)
        word_weights.scatter_add_(0, maskable_wids, enc_mask_weights[maskable].float())
        word_weights = word_weights / word_token_counts.float().clamp(min=1)
        if word_weights.sum() == 0:
            word_weights = torch.ones(num_words)
    else:
        word_weights = torch.ones(num_words)

    pos_ids = torch.arange(seq_len)
    start_positions = torch.full((num_words,), seq_len, dtype=torch.long)
    end_positions = torch.full((num_words,), -1, dtype=torch.long)
    start_positions.scatter_reduce_(0, maskable_wids, pos_ids[maskable], reduce="amin", include_self=True)
    end_positions.scatter_reduce_(0, maskable_wids, pos_ids[maskable], reduce="amax", include_self=True)

    word_order = torch.multinomial(word_weights, num_words, replacement=False)
    target = max(1, int(mlm_prob * seq_len))
    selected = torch.zeros(seq_len, dtype=torch.bool)
    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    mode_pos_ids = torch.full((seq_len,), -1, dtype=torch.long)
    adaptive_stats = torch.zeros(4, dtype=torch.float32)
    selected_count = 0
    span_idx = 0

    for wid in word_order.tolist():
        if selected_count >= target:
            break

        start_pos = start_positions[wid].item()
        end_pos = end_positions[wid].item()
        if start_pos > end_pos:
            continue
        positions = torch.arange(start_pos, end_pos + 1)

        word_key = word_key_ids[wid].item()
        p_wwm = _adaptive_word_wwm_probability(word_key, adaptive_word_stats, args, epoch)
        use_wwm = torch.rand(()) < p_wwm
        adaptive_stats[0] += p_wwm
        adaptive_stats[1] += 1.0

        if use_wwm:
            chosen = positions
            mode_id = 1
        else:
            if len(positions) == 1:
                chosen = positions
            else:
                chosen = positions[torch.randint(len(positions), (1,)).item()].view(1)
            mode_id = 0
            adaptive_stats[3] += 1.0
        if use_wwm:
            adaptive_stats[2] += 1.0

        selected[chosen] = True
        span_ids[chosen] = span_idx
        mode_pos_ids[chosen] = mode_id
        selected_count += len(chosen)
        span_idx += 1

    return selected, span_ids, mode_pos_ids, word_key_ids_per_pos, adaptive_stats


def _adaptive_wwm_start_inside_word_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights,
                                                 mlm_prob, tokenizer, use_weights,
                                                 adaptive_word_stats, args, epoch):
    """
    Word-level adaptive WWM/start/inside masking.

    For each sampled word, choose:
      1 = WWM: mask all subwords in the word
      2 = start: mask only the first subword
      3 = inside: mask one non-start subword
    """
    word_ids, word_key_ids_per_pos, word_key_ids, word_token_counts = _word_ids_and_key_ids(
        seq_len, pad_token_id, enc, tokenizer
    )
    num_words = len(word_key_ids)
    if num_words <= 0:
        return (
            torch.zeros(seq_len, dtype=torch.bool),
            torch.full((seq_len,), -1, dtype=torch.long),
            torch.full((seq_len,), -1, dtype=torch.long),
            word_key_ids_per_pos,
            torch.zeros(4, dtype=torch.float32),
        )

    maskable = word_ids >= 0
    maskable_wids = word_ids[maskable]

    if use_weights:
        word_weights = torch.zeros(num_words)
        word_weights.scatter_add_(0, maskable_wids, enc_mask_weights[maskable].float())
        word_weights = word_weights / word_token_counts.float().clamp(min=1)
        if word_weights.sum() == 0:
            word_weights = torch.ones(num_words)
    else:
        word_weights = torch.ones(num_words)

    pos_ids = torch.arange(seq_len)
    start_positions = torch.full((num_words,), seq_len, dtype=torch.long)
    end_positions = torch.full((num_words,), -1, dtype=torch.long)
    start_positions.scatter_reduce_(0, maskable_wids, pos_ids[maskable], reduce="amin", include_self=True)
    end_positions.scatter_reduce_(0, maskable_wids, pos_ids[maskable], reduce="amax", include_self=True)
    mode_probs = _adaptive_word_wwm_start_inside_probs_for_keys(
        word_key_ids, adaptive_word_stats, args, epoch
    )
    single_word = word_token_counts <= 1
    if single_word.any():
        mode_probs[single_word, 2] = 0.0
        mode_probs[single_word] = mode_probs[single_word] / mode_probs[single_word].sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)

    word_order = torch.multinomial(word_weights, num_words, replacement=False)
    target = max(1, int(mlm_prob * seq_len))
    selected = torch.zeros(seq_len, dtype=torch.bool)
    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    mode_pos_ids = torch.full((seq_len,), -1, dtype=torch.long)
    adaptive_stats = torch.zeros(4, dtype=torch.float32)
    selected_count = 0
    span_idx = 0

    for wid in word_order.tolist():
        if selected_count >= target:
            break

        start_pos = start_positions[wid].item()
        end_pos = end_positions[wid].item()
        if start_pos > end_pos:
            continue

        probs = mode_probs[wid]

        mode = torch.multinomial(probs, 1).item()
        adaptive_stats[0] += probs[0].item()
        adaptive_stats[1] += 1.0

        if mode == 0:
            positions = torch.arange(start_pos, end_pos + 1)
            chosen = positions
            mode_id = 1
            adaptive_stats[2] += 1.0
        elif mode == 1:
            chosen = torch.tensor([start_pos], dtype=torch.long)
            mode_id = 2
            adaptive_stats[3] += 1.0
        else:
            inside_len = end_pos - start_pos
            chosen = torch.tensor(
                [start_pos + 1 + torch.randint(inside_len, (1,)).item()],
                dtype=torch.long,
            )
            mode_id = 3
            adaptive_stats[3] += 1.0

        selected[chosen] = True
        span_ids[chosen] = span_idx
        mode_pos_ids[chosen] = mode_id
        selected_count += len(chosen)
        span_idx += 1

    return selected, span_ids, mode_pos_ids, word_key_ids_per_pos, adaptive_stats


def mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15, mask_replace_prob=0.8,
               random_replace_prob=0.1, span_masking="token", span_len=3,
               span_geo_p=1/3, span_stats_mode="average",
               morph_inside_weight=5.0, morph_start_weight=1.0,
               wwm_token_mix_wwm_prob=0.80,
               adaptive_wwm_token_wwm_prob=None,
               adaptive_word_stats=None, adaptive_epoch=0, adaptive_args=None,
               wwm_mix_whole_prob=0.80, wwm_mix_inside_prob=0.15,
               wwm_mix_start_prob=0.05):
    """
    span_masking:
      "token"      - original per-token independent masking
      "morph_token" - token masking biased toward inside-word subword tokens
      "span_fixed" - contiguous spans of fixed length (span_len)
      "span_geo"   - contiguous spans with geometrically sampled length (span_geo_p)
      "wwm"        - whole word masking
      "wwm_token_mix" - per-sequence random choice between WWM and token masking
      "adaptive_wwm_token" - like wwm_token_mix, but with a training-updated WWM probability
      "adaptive_wwm_token_word" - choose WWM/token per selected word from per-word statistics
      "adaptive_wwm_start_inside_word" - choose WWM/start/inside per selected word
      "wwm_mix"    - WWM sampling, then mixed whole/inside/start corruption
      "wwm_span_fixed" / "wwm_span_geo" - contiguous spans over complete words

    When mask_weights is None (classic MLM, no AMLM): span starts are sampled uniformly.
    When mask_weights is provided (AMLM): span starts are biased by sliding-window difficulty.

    span_stats_mode (only relevant when span_masking != "token"):
      "average" - span loss is split equally across tokens in the span (original behaviour)
      "exact"   - each token records its own individual loss (requires per-token loss at update time)
    """
    amlm = mask_weights is not None
    if mask_weights is None:
        mask_weights = torch.full((tokenizer.vocab_size,), mlm_prob, device=batch["input_ids"].device)

    masked_batch = batch.copy()
    # Store original ids so stats update can do exact attribution
    masked_batch["original_ids"] = batch["input_ids"].clone()

    batch_size = len(batch["input_ids"])
    seq_len_0 = len(batch["input_ids"][0])
    span_ids_batch = torch.full((batch_size, seq_len_0), -1, dtype=torch.long)
    mask_mode_ids = torch.full((batch_size,), -1, dtype=torch.long)
    mask_mode_pos_ids_batch = torch.full((batch_size, seq_len_0), -1, dtype=torch.long)
    word_key_ids_batch = torch.full((batch_size, seq_len_0), -1, dtype=torch.long)
    # columns: p_wwm_sum, prob_count, chosen_wwm_units, chosen_token_units, masked_tokens, nonpad_tokens
    adaptive_stats_batch = torch.zeros((batch_size, 6), dtype=torch.float32)

    for i in range(batch_size):
        seq_len = len(batch["input_ids"][i])
        adaptive_stats_batch[i, 5] = (batch["input_ids"][i] != tokenizer.pad_token_id).sum().item()

        enc_mask_weights = mask_weights[batch["input_ids"][i]]
        enc_mask_weights[batch["input_ids"][i] == tokenizer.pad_token_id] = 0

        cur_span_masking = span_masking
        if span_masking in ("wwm_token_mix", "adaptive_wwm_token"):
            wwm_prob = wwm_token_mix_wwm_prob
            if span_masking == "adaptive_wwm_token" and adaptive_wwm_token_wwm_prob is not None:
                wwm_prob = adaptive_wwm_token_wwm_prob
            cur_span_masking = "wwm" if torch.rand(()) < wwm_prob else "token"
            adaptive_stats_batch[i, 0] = float(wwm_prob)
            adaptive_stats_batch[i, 1] = 1.0
            if cur_span_masking == "wwm":
                adaptive_stats_batch[i, 2] = 1.0
            else:
                adaptive_stats_batch[i, 3] = 1.0

        if cur_span_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word"):
            mask_mode_ids[i] = 5
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            if cur_span_masking == "adaptive_wwm_token_word":
                fn = _adaptive_wwm_token_word_mask_indices
            else:
                fn = _adaptive_wwm_start_inside_word_mask_indices
            enc_mask_mask, span_ids_batch[i], mask_mode_pos_ids_batch[i], word_key_ids_batch[i], word_adaptive_stats = (
                fn(
                    seq_len, tokenizer.pad_token_id, enc, norm_w,
                    mlm_prob, tokenizer, use_weights=amlm,
                    adaptive_word_stats=adaptive_word_stats,
                    args=adaptive_args,
                    epoch=adaptive_epoch,
                )
            )
            adaptive_stats_batch[i, :4] = word_adaptive_stats
        elif cur_span_masking in ("token", "morph_token"):
            mask_mode_ids[i] = 0
            # Original logic: identical to train_mask_original.py
            if cur_span_masking == "morph_token":
                enc_mask_mask = _morph_token_mask_indices(
                    seq_len, tokenizer.pad_token_id, batch["input_ids"][i],
                    enc_mask_weights.float(), mlm_prob, tokenizer,
                    morph_inside_weight, morph_start_weight
                )
            else:
                enc_mask_weights_norm = mlm_prob * seq_len * enc_mask_weights / enc_mask_weights.sum()
                enc_mask_mask = torch.rand(seq_len) < enc_mask_weights_norm

            rand = torch.rand(seq_len)
            to_mask = rand < mask_replace_prob
            to_replace = (rand > mask_replace_prob) & (rand < mask_replace_prob + random_replace_prob)
            to_mask = to_mask & enc_mask_mask
            to_replace = to_replace & enc_mask_mask

            randoms = torch.randint(0, tokenizer.vocab_size, (seq_len,))
            enc = batch["input_ids"][i]
            enc[to_mask] = tokenizer.mask_token_id
            enc[to_replace] = randoms[to_replace]
            masked_batch["input_ids"][i] = enc

            labels = batch["labels"][i]
            labels[~enc_mask_mask] = -100
            masked_batch["labels"][i] = labels
            adaptive_stats_batch[i, 4] = enc_mask_mask.sum().item()
            continue
        elif cur_span_masking == "wwm":
            mask_mode_ids[i] = 1
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            enc_mask_mask, span_ids_batch[i] = _wwm_mask_indices(
                seq_len, tokenizer.pad_token_id, enc, norm_w,
                mlm_prob, tokenizer, use_weights=amlm
            )
            word_key_ids_batch[i] = _word_ids_and_key_ids(seq_len, tokenizer.pad_token_id, enc, tokenizer)[1]
            mask_mode_pos_ids_batch[i][enc_mask_mask] = 1
        elif cur_span_masking == "wwm_mix":
            mask_mode_ids[i] = 2
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            enc_mask_mask, span_ids_batch[i] = _wwm_mix_mask_indices(
                seq_len, tokenizer.pad_token_id, enc, norm_w,
                mlm_prob, tokenizer, use_weights=amlm,
                whole_prob=wwm_mix_whole_prob,
                inside_prob=wwm_mix_inside_prob,
                start_prob=wwm_mix_start_prob
            )
        elif cur_span_masking in ("wwm_span_fixed", "wwm_span_geo"):
            mask_mode_ids[i] = 3
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            span_mode = "fixed" if cur_span_masking == "wwm_span_fixed" else "geo"
            enc_mask_mask, span_ids_batch[i] = _wwm_span_mask_indices(
                seq_len, tokenizer.pad_token_id, enc, norm_w,
                mlm_prob, tokenizer, span_mode, span_len, span_geo_p,
                use_weights=amlm
            )
        elif cur_span_masking == "phrase":
            mask_mode_ids[i] = 6
            enc = batch["input_ids"][i].clone()
            # Use pre-parsed phrase_ids if available, else fall back to token masking
            if "phrase_ids" in batch and batch["phrase_ids"] is not None:
                phrase_ids_enc = batch["phrase_ids"][i]
                enc_mask_mask, span_ids_batch[i] = _phrase_mask_indices(
                    seq_len, tokenizer.pad_token_id, enc, phrase_ids_enc,
                    mlm_prob, tokenizer
                )
            else:
                # Fallback: token masking
                not_special = ~(tokenizer.vocab_is_special[enc] | (enc == tokenizer.pad_token_id))
                prob_tensor = torch.full((seq_len,), mlm_prob)
                prob_tensor[~not_special] = 0.0
                enc_mask_mask = torch.bernoulli(prob_tensor).bool()
                span_ids_batch[i][enc_mask_mask] = torch.arange(enc_mask_mask.sum(), dtype=torch.long)
        else:
            mask_mode_ids[i] = 4
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            span_mode = "fixed" if cur_span_masking == "span_fixed" else "geo"
            # Normalise weights for use in window sampling (keep sum=1 for multinomial)
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            enc_mask_mask, span_ids_batch[i] = _span_mask_indices(
                seq_len, tokenizer.pad_token_id, enc, norm_w,
                mlm_prob, span_mode, span_len, span_geo_p, use_weights=amlm
            )

        if cur_span_masking in ("wwm", "wwm_mix", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word", "wwm_span_fixed", "wwm_span_geo", "span_fixed", "span_geo", "span"):
            enc = _apply_mask_groups(
                enc, enc_mask_mask, span_ids_batch[i],
                tokenizer, mask_replace_prob, random_replace_prob
            )
        else:
            enc = _apply_mask_tokens(enc, enc_mask_mask, tokenizer, mask_replace_prob, random_replace_prob)
        masked_batch["input_ids"][i] = enc

        labels = batch["labels"][i].clone()
        labels[~enc_mask_mask] = -100
        masked_batch["labels"][i] = labels
        adaptive_stats_batch[i, 4] = enc_mask_mask.sum().item()

    masked_batch["span_ids"] = span_ids_batch
    masked_batch["mask_mode_ids"] = mask_mode_ids
    masked_batch["mask_mode_pos_ids"] = mask_mode_pos_ids_batch
    masked_batch["word_key_ids"] = word_key_ids_batch
    masked_batch["adaptive_stats"] = adaptive_stats_batch

    # 记录每个 span 的左右边界位置，用于 SBO（向量化实现）
    # span_boundaries: (batch, seq_len, 2)
    batch_size2, seq_len2 = span_ids_batch.shape
    span_boundaries = torch.full((batch_size2, seq_len2, 2), -1, dtype=torch.long)
    masked = span_ids_batch >= 0  # (batch, seq_len) 被mask的位置
    if masked.any():
        # 对每个被mask位置，找其span的第一个和最后一个位置
        # 利用 span_ids 相同的位置属于同一个span的特性
        # left_edge[i,t] = True 表示 t 是某span的起始位置（前一位不同或t=0）
        sids = span_ids_batch
        left_edge = torch.zeros_like(masked)
        left_edge[:, 0] = masked[:, 0]
        left_edge[:, 1:] = masked[:, 1:] & ((sids[:, 1:] != sids[:, :-1]) | ~masked[:, :-1])

        right_edge = torch.zeros_like(masked)
        right_edge[:, -1] = masked[:, -1]
        right_edge[:, :-1] = masked[:, :-1] & ((sids[:, :-1] != sids[:, 1:]) | ~masked[:, 1:])

        # 对每个masked位置，向左扫到span起始，向右扫到span结束
        # 用cumsum技巧：在left_edge处+1，累加后同一span内值相同
        # 然后用scatter把起始/结束位置广播回去
        idx = torch.arange(seq_len2, device=span_ids_batch.device).unsqueeze(0).expand(batch_size2, -1)

        # left_boundary: span起始位置的前一个（clamp到0）
        left_starts = idx * left_edge + 0 * ~left_edge  # 非起始位置填0，不影响scatter
        # 用span_ids做group：同一span内left_boundary相同
        # 直接：left_boundary[pos] = (span起始pos) - 1
        # 构建 span起始位置 tensor，再通过span_id索引回每个位置
        # 更简单：对每行用scatter_reduce
        num_spans = int(sids.max().item()) + 1 if masked.any() else seq_len2
        span_table_size = max(num_spans, seq_len2)

        left_bound_pos = torch.where(left_edge, idx, torch.full_like(idx, seq_len2))
        span_left = torch.full((batch_size2, span_table_size), seq_len2, dtype=torch.long)
        span_left.scatter_reduce_(1, sids.clamp(min=0), left_bound_pos, reduce="amin", include_self=True)
        gathered_left = span_left.gather(1, sids.clamp(min=0))
        left_b = (gathered_left - 1).clamp(min=0)

        right_bound_pos = torch.where(right_edge, idx, torch.full_like(idx, -1))
        span_right = torch.full((batch_size2, span_table_size), -1, dtype=torch.long)
        span_right.scatter_reduce_(1, sids.clamp(min=0), right_bound_pos, reduce="amax", include_self=True)
        gathered_right = span_right.gather(1, sids.clamp(min=0))
        right_b = (gathered_right + 1).clamp(max=seq_len2 - 1)

        span_boundaries[:, :, 0] = torch.where(masked, left_b, torch.full_like(left_b, -1))
        span_boundaries[:, :, 1] = torch.where(masked, right_b, torch.full_like(right_b, -1))
    masked_batch["span_boundaries"] = span_boundaries

    return masked_batch


def get_batch_accuracy(logits, labels, prev_stats):
    vocab_size = logits.shape[-1]

    # Mask out ignored labels
    mask = labels != -100
    labels = labels[mask]
    preds = logits.argmax(dim=-1)[mask]

    # Get match mask
    correct_mask = preds == labels

    # Use bincount on correct and incorrect
    dtype = prev_stats['correct'].dtype
    correct_counts = torch.bincount(labels[correct_mask], minlength=vocab_size).to(dtype)
    incorrect_counts = torch.bincount(labels[~correct_mask], minlength=vocab_size).to(dtype)

    prev_stats['correct'] += correct_counts
    prev_stats['incorrect'] += incorrect_counts

    return prev_stats


def get_batch_accuracy_average(logits, labels, span_ids, prev_stats):
    """
    Span-average stats attribution for hard (accuracy) mode.
    Each span contributes a single averaged correct/incorrect signal split equally
    across all tokens in the span, so a span of mixed correct/incorrect predictions
    does not unfairly bias any individual token's difficulty estimate.
    """
    vocab_size = logits.shape[-1]
    preds = logits.argmax(dim=-1)  # (batch, seq)

    for seq_preds, seq_labels, seq_span_ids in zip(preds, labels, span_ids):
        active = seq_labels != -100
        if not active.any():
            continue

        active_span_ids = seq_span_ids[active]   # span id per active position
        active_labels   = seq_labels[active]
        correct         = (seq_preds[active] == active_labels).float()

        num_spans = int(active_span_ids.max().item()) + 1

        # sum correct count and token count per span
        span_correct = torch.zeros(num_spans, device=correct.device).scatter_add_(0, active_span_ids, correct)
        span_count   = torch.zeros(num_spans, device=correct.device).scatter_add_(0, active_span_ids, torch.ones_like(correct))

        # broadcast span-level fraction back to each active position
        span_frac         = torch.nan_to_num(span_correct / span_count)  # shape: (num_spans,)
        pos_correct_frac  = span_frac[active_span_ids]                   # shape: (num_active,)
        pos_incorrect_frac = 1.0 - pos_correct_frac

        dtype = prev_stats['correct'].dtype
        prev_stats['correct'].scatter_add_(0, active_labels, pos_correct_frac.to(dtype))
        prev_stats['incorrect'].scatter_add_(0, active_labels, pos_incorrect_frac.to(dtype))

    return prev_stats


def get_batch_accuracy_exact(logits, labels, prev_stats):
    """
    Exact per-token stats attribution for span masking.
    Each masked token updates its own correct/incorrect count independently,
    avoiding the averaging artifact that occurs when span stats are split equally.
    Functionally identical to get_batch_accuracy for token-level masking,
    but kept separate for clarity and future divergence.
    """
    vocab_size = logits.shape[-1]

    mask = labels != -100
    flat_labels = labels[mask]
    flat_preds = logits.argmax(dim=-1)[mask]

    correct_mask = flat_preds == flat_labels
    dtype = prev_stats['correct'].dtype
    prev_stats['correct'] += torch.bincount(flat_labels[correct_mask], minlength=vocab_size).to(dtype)
    prev_stats['incorrect'] += torch.bincount(flat_labels[~correct_mask], minlength=vocab_size).to(dtype)

    return prev_stats

def update_mask_weights(mask_weights, mask_stats, mlm_prob=0.15):
    correct_pct = (mask_stats['correct'] + 0.5) / (mask_stats['incorrect'] + mask_stats['correct'] + 1)
    # if there are no instances, treated as 50% accuracy
    # things that never show up will go to 7.5% mlm_prob (seems fine?)

    new_weights = mlm_prob - (correct_pct * mlm_prob)
    mask_weights = 0.2 * mask_weights + 0.8 * new_weights

    mask_weights = mask_weights.clamp(0.005)
    mask_weights = mlm_prob * mask_weights.shape[0] * mask_weights / mask_weights.sum() # normalize back to mlm_prob

    return mask_weights

def compute_sbo_loss(hidden_states, labels, span_boundaries, sbo_head, position_embeddings):
    """
    Span Boundary Objective (SpanBERT), 向量化实现。
    hidden_states:   (batch, seq_len, hidden_size)
    labels:          (batch, seq_len)
    span_boundaries: (batch, seq_len, 2)
    sbo_head:        Linear(3*hidden_size → vocab_size)
    position_embeddings: (max_span_len, hidden_size)
    """
    # 找出所有有效的 masked 位置（span_boundaries 都 >= 0）
    valid = (labels != -100) & (span_boundaries[:, :, 0] >= 0) & (span_boundaries[:, :, 1] >= 0)
    if not valid.any():
        return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

    # 取出 batch index 和 seq index
    b_idx, pos_idx = valid.nonzero(as_tuple=True)  # (N,)

    left_b  = span_boundaries[b_idx, pos_idx, 0]   # (N,)
    right_b = span_boundaries[b_idx, pos_idx, 1]   # (N,)

    h_left  = hidden_states[b_idx, left_b]          # (N, hidden_size)
    h_right = hidden_states[b_idx, right_b]         # (N, hidden_size)

    # 相对位置编码（clamp 到 max_span_len-1）
    rel_pos = (pos_idx - left_b).clamp(max=position_embeddings.shape[0] - 1)  # (N,)
    h_pos   = position_embeddings[rel_pos]           # (N, hidden_size)

    h = torch.cat([h_left, h_right, h_pos], dim=-1) # (N, 3*hidden_size)
    logits  = sbo_head(h)                            # (N, vocab_size)
    targets = labels[b_idx, pos_idx]                 # (N,)

    return torch.nn.functional.cross_entropy(logits, targets)


def compute_wwm_pos_loss(hidden_states, labels, original_ids, lm_head,
                         position_embeddings, vocab_is_word_start,
                         vocab_is_special=None, position_scheme="binary"):
    """
    Lightweight word-internal position auxiliary loss.

    It predicts masked tokens from the normal hidden state plus a small
    word-internal position embedding:

    binary:
      0 = token starts a word (SentencePiece "▁")
      1 = token is inside a word

    single_start_middle_end:
      0 = single-token word
      1 = start token of a multi-token word
      2 = middle token of a multi-token word
      3 = end token of a multi-token word

    This is used only during training, similar to SBO, so saved checkpoints
    remain standard masked-LM checkpoints for evaluation.
    """
    valid = labels != -100
    if not valid.any():
        return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

    vocab_is_word_start = vocab_is_word_start.to(device=hidden_states.device)
    is_word_start = vocab_is_word_start[original_ids]

    if position_scheme == "binary":
        pos_ids = torch.where(is_word_start, torch.zeros_like(labels), torch.ones_like(labels))
    else:
        if vocab_is_special is None:
            vocab_is_special = torch.zeros_like(vocab_is_word_start, dtype=torch.bool)
        vocab_is_special = vocab_is_special.to(device=hidden_states.device)
        is_special = vocab_is_special[original_ids]
        maskable = ~is_special

        # Build per-sequence word ids from SentencePiece word-start markers.
        # The input text normally starts content words with ▁, but force the
        # first maskable token to a word start for robustness.
        forced_start = torch.zeros_like(maskable)
        first_maskable = maskable.float().argmax(dim=1)
        has_maskable = maskable.any(dim=1)
        forced_start[torch.arange(maskable.shape[0], device=maskable.device), first_maskable] = has_maskable
        is_word_start = (is_word_start | forced_start) & maskable

        word_ids_raw = is_word_start.long().cumsum(dim=1) - 1
        word_ids = torch.where(maskable, word_ids_raw, torch.full_like(word_ids_raw, -1))

        batch_size, seq_len = labels.shape
        row_ids = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(batch_size, seq_len)
        col_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, seq_len)
        valid_word = word_ids >= 0
        group_ids = row_ids[valid_word] * seq_len + word_ids[valid_word]
        num_groups = batch_size * seq_len

        group_count = torch.zeros(num_groups, device=hidden_states.device, dtype=torch.long)
        group_start = torch.full((num_groups,), seq_len, device=hidden_states.device, dtype=torch.long)
        group_count.scatter_add_(0, group_ids, torch.ones_like(group_ids, dtype=torch.long))
        group_start.scatter_reduce_(0, group_ids, col_ids[valid_word], reduce="amin", include_self=True)

        pos_ids = torch.zeros_like(labels)
        active_group_ids = row_ids[valid_word] * seq_len + word_ids[valid_word]
        lengths = group_count[active_group_ids]
        offsets = col_ids[valid_word] - group_start[active_group_ids]

        pos_valid = torch.full_like(offsets, 2)
        pos_valid[lengths == 1] = 0
        pos_valid[(lengths > 1) & (offsets == 0)] = 1
        pos_valid[(lengths > 1) & (offsets == lengths - 1)] = 3
        pos_ids[valid_word] = pos_valid

    h = hidden_states[valid] + position_embeddings(pos_ids[valid])
    logits = lm_head(h)
    targets = labels[valid]
    return torch.nn.functional.cross_entropy(logits.float(), targets)


def get_batch_accuracy_soft_average(masked_inputs, logits, labels, span_ids, prev_stats):
    """
    Span-average stats attribution for soft (loss) mode.
    Only [MASK] positions contribute to span loss averages (consistent with original behaviour).
    The averaged loss is then attributed to all [MASK] tokens in the span.
    """
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        per_token_loss = per_token_loss.view(masked_inputs.shape)

        for seq_inputs, seq_loss, seq_labels, seq_span_ids in zip(masked_inputs, per_token_loss, labels, span_ids):
            # only [MASK] positions, consistent with original get_batch_accuracy_soft
            active = seq_inputs == 4  # mask_token_id
            if not active.any():
                continue

            active_span_ids = seq_span_ids[active]
            active_labels   = seq_labels[active]
            active_loss     = seq_loss[active].float()

            num_spans = int(active_span_ids.max().item()) + 1

            span_loss_sum = torch.zeros(num_spans, device=active_loss.device).scatter_add_(0, active_span_ids, active_loss)
            span_count    = torch.zeros(num_spans, device=active_loss.device).scatter_add_(0, active_span_ids, torch.ones_like(active_loss))

            span_avg_loss = torch.nan_to_num(span_loss_sum / span_count)
            pos_avg_loss  = span_avg_loss[active_span_ids]

            dtype = prev_stats['loss'].dtype
            prev_stats['loss'].scatter_add_(0, active_labels, pos_avg_loss.to(dtype))
            prev_stats['total'].scatter_add_(0, active_labels, torch.ones_like(pos_avg_loss).to(dtype))

        return prev_stats


def get_batch_accuracy_soft(masked_inputs, logits, labels, prev_stats):
    with torch.no_grad():
        vocab_size = logits.shape[-1]

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        loss = loss.view(masked_inputs.shape)

        # Mask out ignored labels and replaced tokens
        mask = masked_inputs == 4 # mask_token_id

        loss = loss[mask]
        labels = labels[mask]

        prev_stats['loss'].scatter_add_(0, labels, loss)
        prev_stats['total'].scatter_add_(0, labels, torch.ones_like(loss))

        return prev_stats


def get_batch_accuracy_soft_exact(masked_inputs, logits, labels, prev_stats):
    """
    Exact per-token loss attribution for span masking, soft stats mode.
    Only [MASK] positions are used (consistent with token-mode original behaviour).
    """
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss = loss.view(masked_inputs.shape)

        mask = masked_inputs == 4  # mask_token_id, same as original

        loss = loss[mask]
        active_labels = labels[mask]

        prev_stats['loss'].scatter_add_(0, active_labels, loss)
        prev_stats['total'].scatter_add_(0, active_labels, torch.ones_like(loss))

        return prev_stats
    
def update_mask_weights_soft(mask_weights, mask_stats, mlm_prob=0.15):
    max_loss = torch.log(torch.tensor(mask_weights.shape[0])).item() # e.g. log(40000) = 10.5966, uniform guess
    avgs = mask_stats['loss'] / mask_stats['total']
    norm_loss = avgs / max_loss

    # min-max normalize and invert
    norm_loss = 1 - (norm_loss - norm_loss.min()) / (norm_loss.max() - norm_loss.min())

    new_weights = mlm_prob - (norm_loss * mlm_prob)
    mask_weights = 0.2 * mask_weights + 0.8 * new_weights

    mask_weights = mask_weights.clamp(0.005)
    mask_weights = mlm_prob * mask_weights.shape[0] * mask_weights / mask_weights.sum()

    return mask_weights

def split_batch(batch, args):
    minibatch_size = args.batch_size // args.grad_acc
    if len(batch["input_ids"]) == minibatch_size:
        return [batch]

    batches = []
    for i in range(0, len(batch["input_ids"]), minibatch_size):
        minibatch = {}
        for key in batch.keys():
            minibatch[key] = batch[key][i:i+minibatch_size]
        batches.append(minibatch)
    return batches

def move_dict_to_cuda(d):
    return {key: value.to(device="cuda:0") for key, value in d.items()}

def move_dict_to_cuda_bf16(d):
    return {key: value.to(dtype=torch.bfloat16, device="cuda:0") for key, value in d.items()}

def get_amp_dtype():
    # V100 does not support bfloat16; use float16 instead
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def reset_stats(mask_stats):
    mask_stats = {
        'correct': torch.zeros_like(mask_stats['correct']), 
        'incorrect': torch.zeros_like(mask_stats['incorrect']),
        'loss': torch.zeros_like(mask_stats['loss']) + torch.log(torch.tensor(mask_stats['loss'].shape[0])).item(),
        'total': torch.ones_like(mask_stats['total'])
        }
    return mask_stats

def calculate_num_words(examples):
    return {'num_words': [sum(len(examples["text"][i]) for i in range(len(examples["text"])))]}

def calculate_num_tokens(examples):
    return {'num_tokens': [sum(len(examples["input_ids"][i]) for i in range(len(examples["input_ids"])))]}

def calculate_total_steps(args):
    def calc_exs_per_epoch(tokens_per_1000, max_seq_len):
        return sum([t // max_seq_len for t in tokens_per_1000])

    
    exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, args.init_max_seq_len)
    batches_per_epoch = math.ceil(exs_per_epoch / args.batch_size)
    cur_epoch = 0
    if len(args.max_seq_len) == 0:
        return batches_per_epoch * args.epochs
    else:
        total_steps = 0
        batch_size = args.batch_size
        prev_seq_len = args.init_max_seq_len
        for epoch_num, seq_len in args.max_seq_len:
            total_steps = total_steps + batches_per_epoch * (epoch_num - cur_epoch)

            len_ratio = prev_seq_len / seq_len
            batch_size = int(batch_size * len_ratio)

            exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, seq_len)
            batches_per_epoch = math.ceil(exs_per_epoch / batch_size)
            cur_epoch = epoch_num
            prev_seq_len = seq_len
    
        total_steps = total_steps + batches_per_epoch * (args.epochs - cur_epoch)
        return total_steps


def is_step(step_type: str, global_step: int, args):
    # step_arg = args.logging_steps, args.save_steps, or args.eval_steps
    step_arg = getattr(args, f'{step_type}_steps')

    if args.all_checkpoints:
        if global_step in args.checkpoints:
            return True
    else:
        if global_step % step_arg == 0 and global_step != 0:
            return True
        
    return False


def init_plateau_switch_state():
    return {
        "switched": False,
        "switch_step": -1,
        "switch_epoch": -1.0,
        "prev_loss": None,
        "bad_windows": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
    }


def maybe_update_plateau_switch(state, args, global_step, epoch_float):
    if state["loss_count"] == 0:
        return False

    cur_loss = state["loss_sum"] / state["loss_count"]
    state["loss_sum"] = 0.0
    state["loss_count"] = 0

    if state["prev_loss"] is None:
        state["prev_loss"] = cur_loss
        print(
            f"[WWM plateau switch] step={global_step}, epoch={epoch_float:.2f}, "
            f"wwm_loss={cur_loss:.4f}, improvement=NA, bad_windows=0",
            flush=True,
        )
        return False

    improvement = state["prev_loss"] - cur_loss
    state["prev_loss"] = cur_loss
    can_switch = epoch_float >= args.plateau_switch_min_epochs
    force_switch = args.plateau_switch_max_epochs > 0 and epoch_float >= args.plateau_switch_max_epochs

    if can_switch and improvement < args.plateau_switch_threshold:
        state["bad_windows"] += 1
    else:
        state["bad_windows"] = 0

    should_switch = force_switch or (can_switch and state["bad_windows"] >= args.plateau_switch_patience)
    print(
        f"[WWM plateau switch] step={global_step}, epoch={epoch_float:.2f}, "
        f"wwm_loss={cur_loss:.4f}, improvement={improvement:.4f}, "
        f"bad_windows={state['bad_windows']}/{args.plateau_switch_patience}, "
        f"switch={should_switch}",
        flush=True,
    )
    if should_switch:
        state["switched"] = True
        state["switch_step"] = global_step
        state["switch_epoch"] = epoch_float
    return should_switch


def update_wwm_mix_probs(args, epoch):
    """Update active WWM-mix probabilities for the current epoch."""
    if args.wwm_mix_schedule != "linear":
        args._active_wwm_mix_whole_prob = args.wwm_mix_whole_prob
        args._active_wwm_mix_inside_prob = args.wwm_mix_inside_prob
        args._active_wwm_mix_start_prob = args.wwm_mix_start_prob
        return

    # Interpolate only across the phase where wwm_mix is active.
    if args.token_epochs > 0:
        phase_start = args.token_epochs
        phase_len = max(args.epochs - args.token_epochs, 1)
        phase_epoch = max(epoch - phase_start, 0)
    elif args.span_epochs > 0:
        phase_len = max(args.span_epochs, 1)
        phase_epoch = min(epoch, phase_len - 1)
    else:
        phase_len = max(args.epochs, 1)
        phase_epoch = min(epoch, phase_len - 1)

    denom = max(phase_len - 1, 1)
    t = min(max(phase_epoch / denom, 0.0), 1.0)

    args._active_wwm_mix_whole_prob = (
        args.wwm_mix_start_whole_prob
        + t * (args.wwm_mix_end_whole_prob - args.wwm_mix_start_whole_prob)
    )
    args._active_wwm_mix_inside_prob = (
        args.wwm_mix_start_inside_prob
        + t * (args.wwm_mix_end_inside_prob - args.wwm_mix_start_inside_prob)
    )
    args._active_wwm_mix_start_prob = (
        args.wwm_mix_start_start_prob
        + t * (args.wwm_mix_end_start_prob - args.wwm_mix_start_start_prob)
    )


def init_adaptive_wwm_token_state(args, tokenizer):
    max_loss = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
    initial_p = min(
        max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
        args.adaptive_wwm_token_max_wwm_prob,
    )
    return {
        "wwm_loss_ema": max_loss,
        "token_loss_ema": max_loss,
        "wwm_prev_loss_ema": max_loss,
        "token_prev_loss_ema": max_loss,
        "wwm_improvement": 0.0,
        "token_improvement": 0.0,
        "wwm_count": 0,
        "token_count": 0,
        "wwm_prob": initial_p,
        "max_loss": max_loss,
    }


def _adaptive_epsilon(args, epoch):
    if args.epochs <= 1:
        return args.adaptive_wwm_token_epsilon_end
    t = min(max(epoch / (args.epochs - 1), 0.0), 1.0)
    return (
        args.adaptive_wwm_token_epsilon_start
        + t * (args.adaptive_wwm_token_epsilon_end - args.adaptive_wwm_token_epsilon_start)
    )


def update_adaptive_wwm_token_state(state, logits, labels, mode_ids, args, epoch):
    """
    Update global WWM-vs-token diagnostics and the next-step WWM probability.

    This first adaptive version intentionally keeps the statistics coarse:
    all token-level positions are merged into one token objective. That makes
    the signal less noisy than splitting into start/middle/end on small data.
    """
    with torch.no_grad():
        labels = labels.to(device=logits.device)
        active = labels != -100
        if not active.any():
            return state

        loss = torch.zeros(labels.shape, dtype=torch.float32, device=logits.device)
        loss[active] = torch.nn.functional.cross_entropy(
            logits.float()[active],
            labels[active],
            reduction="none",
        )

        mode_ids = mode_ids.to(device=logits.device)
        row_modes = mode_ids[:, None].expand_as(labels)
        alpha = args.adaptive_wwm_token_ema_alpha

        for name, mode_id in (("token", 0), ("wwm", 1)):
            mode_mask = active & (row_modes == mode_id)
            if not mode_mask.any():
                continue

            cur_loss = loss[mode_mask].mean().item()
            loss_key = f"{name}_loss_ema"
            prev_key = f"{name}_prev_loss_ema"
            imp_key = f"{name}_improvement"
            count_key = f"{name}_count"

            old_loss = state[loss_key]
            if state[count_key] == 0:
                new_loss = cur_loss
            else:
                new_loss = (1.0 - alpha) * old_loss + alpha * cur_loss

            state[prev_key] = old_loss
            state[loss_key] = new_loss
            state[imp_key] = old_loss - new_loss
            state[count_key] += int(mode_mask.sum().item())

        # During warmup, keep the fixed prior but still collect statistics.
        if epoch < args.adaptive_wwm_token_warmup_epochs:
            state["wwm_prob"] = min(
                max(args.adaptive_wwm_token_initial_wwm_prob, args.adaptive_wwm_token_min_wwm_prob),
                args.adaptive_wwm_token_max_wwm_prob,
            )
            return state

        max_loss = max(state["max_loss"], 1e-8)
        wwm_score = (
            state["wwm_loss_ema"] / max_loss
            + args.adaptive_wwm_token_beta * max(state["wwm_improvement"], 0.0) / max_loss
        )
        token_score = (
            state["token_loss_ema"] / max_loss
            + args.adaptive_wwm_token_beta
            * args.adaptive_wwm_token_token_improvement_discount
            * max(state["token_improvement"], 0.0)
            / max_loss
        )

        temp = max(args.adaptive_wwm_token_temperature, 1e-6)
        scores = torch.tensor([wwm_score, token_score], dtype=torch.float32) / temp
        probs = torch.softmax(scores, dim=0)
        adaptive_p = probs[0].item()

        epsilon = _adaptive_epsilon(args, epoch)
        mixed_p = (1.0 - epsilon) * adaptive_p + epsilon * 0.5
        state["wwm_prob"] = min(
            max(mixed_p, args.adaptive_wwm_token_min_wwm_prob),
            args.adaptive_wwm_token_max_wwm_prob,
        )

    return state


def init_adaptive_word_stats(args, tokenizer):
    max_loss = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
    return {
        "wwm_loss_ema": torch.full((tokenizer.vocab_size,), max_loss, dtype=torch.float32),
        "token_loss_ema": torch.full((tokenizer.vocab_size,), max_loss, dtype=torch.float32),
        "start_loss_ema": torch.full((tokenizer.vocab_size,), max_loss, dtype=torch.float32),
        "inside_loss_ema": torch.full((tokenizer.vocab_size,), max_loss, dtype=torch.float32),
        "wwm_improvement": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "token_improvement": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "start_improvement": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "inside_improvement": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "wwm_count": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "token_count": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "start_count": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "inside_count": torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        "max_loss": max_loss,
    }


def _update_adaptive_word_mode(stats, mode_name, keys, losses, alpha):
    if len(keys) == 0:
        return

    valid = keys >= 0
    if not valid.any():
        return

    keys = keys[valid].long()
    losses = losses[valid].float()

    active_keys, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    loss_sum = torch.zeros(len(active_keys), dtype=torch.float32)
    loss_count = torch.zeros(len(active_keys), dtype=torch.float32)
    loss_sum.scatter_add_(0, inverse, losses)
    loss_count.scatter_add_(0, inverse, torch.ones_like(losses))

    cur_loss = loss_sum / loss_count.clamp(min=1.0)
    loss_key = f"{mode_name}_loss_ema"
    imp_key = f"{mode_name}_improvement"
    count_key = f"{mode_name}_count"

    old_loss = stats[loss_key][active_keys]
    old_count = stats[count_key][active_keys]
    new_loss = torch.where(old_count == 0, cur_loss, (1.0 - alpha) * old_loss + alpha * cur_loss)

    stats[loss_key][active_keys] = new_loss
    stats[imp_key][active_keys] = old_loss - new_loss
    stats[count_key][active_keys] += loss_count


def update_adaptive_word_stats(stats, logits, labels, span_ids, mode_pos_ids, word_key_ids, args):
    """Update per-word WWM/token EMA losses. Word key is the word-start token id."""
    with torch.no_grad():
        labels = labels.to(device=logits.device)
        active = labels != -100
        if not active.any():
            return

        loss = torch.zeros(labels.shape, dtype=torch.float32, device=logits.device)
        loss[active] = torch.nn.functional.cross_entropy(
            logits.float()[active],
            labels[active],
            reduction="none",
        )

        mode_pos_ids_cpu = mode_pos_ids.detach().cpu()
        word_key_ids_cpu = word_key_ids.detach().cpu()
        span_ids_cpu = span_ids.detach().cpu()
        loss_cpu = loss.detach().cpu()
        labels_cpu = labels.detach().cpu()
        active_cpu = labels_cpu != -100

        token_mask = active_cpu & (mode_pos_ids_cpu == 0)
        _update_adaptive_word_mode(
            stats,
            "token",
            word_key_ids_cpu[token_mask],
            loss_cpu[token_mask],
            args.adaptive_wwm_token_ema_alpha,
        )

        start_mask = active_cpu & (mode_pos_ids_cpu == 2)
        _update_adaptive_word_mode(
            stats,
            "start",
            word_key_ids_cpu[start_mask],
            loss_cpu[start_mask],
            args.adaptive_wwm_token_ema_alpha,
        )

        inside_mask = active_cpu & (mode_pos_ids_cpu == 3)
        _update_adaptive_word_mode(
            stats,
            "inside",
            word_key_ids_cpu[inside_mask],
            loss_cpu[inside_mask],
            args.adaptive_wwm_token_ema_alpha,
        )

        wwm_mask = active_cpu & (mode_pos_ids_cpu == 1) & (span_ids_cpu >= 0) & (word_key_ids_cpu >= 0)
        if wwm_mask.any():
            batch_size, seq_len = labels_cpu.shape
            row_ids = torch.arange(batch_size).unsqueeze(1).expand(batch_size, seq_len)
            group_ids = row_ids[wwm_mask] * seq_len + span_ids_cpu[wwm_mask]
            num_groups = batch_size * seq_len

            group_loss_sum = torch.zeros(num_groups, dtype=torch.float32)
            group_count = torch.zeros(num_groups, dtype=torch.float32)
            group_key = torch.full((num_groups,), -1, dtype=torch.long)

            group_loss_sum.scatter_add_(0, group_ids, loss_cpu[wwm_mask].float())
            group_count.scatter_add_(0, group_ids, torch.ones_like(group_ids, dtype=torch.float32))
            group_key.scatter_reduce_(0, group_ids, word_key_ids_cpu[wwm_mask].long(), reduce="amax", include_self=True)

            active_groups = group_count > 0
            wwm_keys = group_key[active_groups]
            wwm_losses = group_loss_sum[active_groups] / group_count[active_groups].clamp(min=1.0)
            _update_adaptive_word_mode(
                stats,
                "wwm",
                wwm_keys,
                wwm_losses,
                args.adaptive_wwm_token_ema_alpha,
            )


def summarize_adaptive_word_stats(stats):
    wwm_seen = stats["wwm_count"] > 0
    token_seen = stats["token_count"] > 0
    start_seen = stats["start_count"] > 0
    inside_seen = stats["inside_count"] > 0
    both_seen = wwm_seen & token_seen
    three_seen = wwm_seen & start_seen & inside_seen

    def mean_or_zero(values, mask):
        return values[mask].mean().item() if mask.any() else 0.0

    return {
        "wwm_seen": int(wwm_seen.sum().item()),
        "token_seen": int(token_seen.sum().item()),
        "start_seen": int(start_seen.sum().item()),
        "inside_seen": int(inside_seen.sum().item()),
        "both_seen": int(both_seen.sum().item()),
        "three_seen": int(three_seen.sum().item()),
        "wwm_loss": mean_or_zero(stats["wwm_loss_ema"], wwm_seen),
        "token_loss": mean_or_zero(stats["token_loss_ema"], token_seen),
        "start_loss": mean_or_zero(stats["start_loss_ema"], start_seen),
        "inside_loss": mean_or_zero(stats["inside_loss_ema"], inside_seen),
        "wwm_imp": mean_or_zero(stats["wwm_improvement"], wwm_seen),
        "token_imp": mean_or_zero(stats["token_improvement"], token_seen),
        "start_imp": mean_or_zero(stats["start_improvement"], start_seen),
        "inside_imp": mean_or_zero(stats["inside_improvement"], inside_seen),
    }



import re as _re


def apply_freeze(model, args):
    if args.freeze_layers:
        freeze_indices = set(int(x.strip()) for x in args.freeze_layers.split(",") if x.strip())
        for name, param in model.named_parameters():
            freeze = False
            if "deberta.embeddings" in name:
                freeze = True
            else:
                m = _re.search(r"deberta\.encoder\.layer\.(\d+)", name)
                if m and int(m.group(1)) in freeze_indices:
                    freeze = True
            if freeze:
                param.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Freeze] Froze encoder layers {sorted(freeze_indices)} + embeddings. "
              f"Frozen={frozen:,}, Trainable={trainable:,}", flush=True)
    elif args.freeze_lower_layers > 0:
        for name, param in model.named_parameters():
            freeze = False
            if "deberta.embeddings" in name:
                freeze = True
            else:
                m = _re.search(r"deberta\.encoder\.layer\.(\d+)", name)
                if m and int(m.group(1)) < args.freeze_lower_layers:
                    freeze = True
            if freeze:
                param.requires_grad = False
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Freeze] Froze encoder layers 0-{args.freeze_lower_layers - 1} + embeddings. "
              f"Frozen={frozen:,}, Trainable={trainable:,}", flush=True)


def train(args, model, tokenizer, train_dataloader, eval_dataloader):
    if args.flops:
        from fvcore.nn import FlopCountAnalysis


    model.train()
    AMP_DTYPE = get_amp_dtype()
    model = model.to(dtype=AMP_DTYPE, device="cuda:0")

    # 初始化 active span masking 模式（forward curriculum 时先强制 token）
    if args.token_epochs > 0 and args.span_masking != "token":
        args._active_span_masking = "token"
    else:
        args._active_span_masking = args.span_masking
    update_wwm_mix_probs(args, 0)

    # SBO 组件（只在 span 模式且 --sbo 开启时使用）
    sbo_head = None
    sbo_position_embeddings = None
    wwm_pos_embeddings = None
    if args.span_masking != "token" and args.sbo:
        max_span_len = 10
        sbo_head = torch.nn.Linear(3 * args.hidden_size, tokenizer.vocab_size, bias=False)
        sbo_position_embeddings = torch.nn.Embedding(max_span_len, args.hidden_size)
        sbo_head = sbo_head.to(dtype=AMP_DTYPE, device="cuda:0")
        sbo_position_embeddings = sbo_position_embeddings.to(dtype=AMP_DTYPE, device="cuda:0")
        all_params = list(model.parameters()) + list(sbo_head.parameters()) + list(sbo_position_embeddings.parameters())
    else:
        all_params = list(model.parameters())

    if args.wwm_pos:
        args._wwm_pos_modes = {mode.strip() for mode in args.wwm_pos_modes.split(",") if mode.strip()}
        wwm_pos_size = 4 if args.wwm_pos_scheme == "single_start_middle_end" else 2
        wwm_pos_embeddings = torch.nn.Embedding(wwm_pos_size, args.hidden_size)
        wwm_pos_embeddings = wwm_pos_embeddings.to(dtype=AMP_DTYPE, device="cuda:0")
        all_params = all_params + list(wwm_pos_embeddings.parameters())
        print(
            f"[WWM position] Enabled word-internal position auxiliary loss, "
            f"scheme={args.wwm_pos_scheme}, classes={wwm_pos_size}, "
            f"weight={args.wwm_pos_weight}, modes={sorted(args._wwm_pos_modes)}",
            flush=True,
        )
    else:
        args._wwm_pos_modes = set()

    if args.lamb:
        optimizer = LAMB(all_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.total_steps//100, num_training_steps=args.total_steps)


    mask_weights = torch.full((tokenizer.vocab_size,), args.mlm_prob).to(device="cuda:0")
    mask_stats = {
        'correct': torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        'incorrect': torch.zeros(tokenizer.vocab_size, dtype=torch.float32),
        'loss': torch.zeros(tokenizer.vocab_size, dtype=torch.float32) + torch.log(torch.tensor(mask_weights.shape[0])).item(),
        'total': torch.ones(tokenizer.vocab_size, dtype=torch.float32)
        }
    mask_stats = move_dict_to_cuda(mask_stats)
    log_file = None
    vocab_list = None
    if args.log_mlm_probs:
        # create parent folder
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        log_file = open(os.path.join(args.output_path, "mlm_probs.csv"), "w")
        # get all vocab as list
        vocab_list = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
        header = ",".join(vocab_list)
        log_file.write(f"step,{header}\n")

    # Resume from checkpoint: fast-forward scheduler and adjust mlm_prob
    resume_step = 0
    if args.resume_from_checkpoint is not None:
        ckpt_name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        resume_step = int(ckpt_name.split("-")[-1])
        print(f"Resuming from checkpoint {args.resume_from_checkpoint}, step {resume_step}", flush=True)
        # Advance scheduler to resume_step without gradient updates
        for _ in range(resume_step):
            scheduler.step()
        # Adjust mlm_prob to reflect decay already applied
        if args.mask_decay > 0:
            args.mlm_prob = max(0.0, args.mlm_prob - args.mask_decay * resume_step / args.total_steps)
        print(f"Restored mlm_prob={args.mlm_prob:.4f}, lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    global_step = resume_step
    print(f"Total steps: {args.total_steps}")
    print(f"Max seq len: {args.init_max_seq_len}")
    print(f"Next seq len: {args.max_seq_len}")

    adaptive_wwm_token_state = None
    adaptive_word_stats = None
    plateau_switch_state = None
    args._active_adaptive_wwm_prob = args.adaptive_wwm_token_initial_wwm_prob
    uses_adaptive_word_stats = (
        args.span_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word")
        or args.transition_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word")
    )
    if args.span_masking == "adaptive_wwm_token":
        adaptive_wwm_token_state = init_adaptive_wwm_token_state(args, tokenizer)
        args._active_adaptive_wwm_prob = adaptive_wwm_token_state["wwm_prob"]
        print(
            f"[Adaptive WWM/token] Enabled: initial_wwm={args._active_adaptive_wwm_prob:.3f}, "
            f"warmup_epochs={args.adaptive_wwm_token_warmup_epochs}, "
            f"clamp=[{args.adaptive_wwm_token_min_wwm_prob:.2f}, {args.adaptive_wwm_token_max_wwm_prob:.2f}], "
            f"token_improvement_discount={args.adaptive_wwm_token_token_improvement_discount:.2f}",
            flush=True,
        )
    elif uses_adaptive_word_stats:
        adaptive_word_stats = init_adaptive_word_stats(args, tokenizer)
        args._adaptive_word_stats_cpu = adaptive_word_stats
        adaptive_word_mode = args.span_masking if args.span_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word") else args.transition_masking
        if adaptive_word_mode == "adaptive_wwm_token_word":
            print(
                f"[Word adaptive WWM/token] Enabled: initial_wwm={args.adaptive_wwm_token_initial_wwm_prob:.3f}, "
                f"warmup_epochs={args.adaptive_wwm_token_warmup_epochs}, "
                f"min_count={args.adaptive_wwm_token_min_count}, "
                f"clamp=[{args.adaptive_wwm_token_min_wwm_prob:.2f}, {args.adaptive_wwm_token_max_wwm_prob:.2f}], "
                f"token_improvement_discount={args.adaptive_wwm_token_token_improvement_discount:.2f}",
                flush=True,
            )
        else:
            print(
                f"[Word adaptive WWM/start/inside] Enabled: initial="
                f"{args.adaptive_wwm_token_initial_wwm_prob:.2f}/"
                f"{args.adaptive_wwm_start_inside_initial_start_prob:.2f}/"
                f"{args.adaptive_wwm_start_inside_initial_inside_prob:.2f}, "
                f"warmup_epochs={args.adaptive_wwm_token_warmup_epochs}, "
                f"min_count={args.adaptive_wwm_token_min_count}, "
                f"wwm_clamp=[{args.adaptive_wwm_token_min_wwm_prob:.2f}, {args.adaptive_wwm_token_max_wwm_prob:.2f}], "
                f"token_improvement_discount={args.adaptive_wwm_token_token_improvement_discount:.2f}",
                flush=True,
            )
    elif args.span_masking == "wwm_token_plateau":
        plateau_switch_state = init_plateau_switch_state()
        check_steps = args.plateau_switch_check_steps or args.mask_update_steps
        print(
            f"[WWM plateau switch] Enabled: start=wwm, then token when WWM loss plateaus; "
            f"min_epochs={args.plateau_switch_min_epochs}, "
            f"max_epochs={args.plateau_switch_max_epochs}, "
            f"patience={args.plateau_switch_patience}, "
            f"threshold={args.plateau_switch_threshold}, "
            f"check_steps={check_steps}",
            flush=True,
        )

    adaptive_prob_log_file = None
    adaptive_word_wwm_prob_log_file = None
    adaptive_word_token_prob_log_file = None
    adaptive_word_start_prob_log_file = None
    adaptive_word_inside_prob_log_file = None
    adaptive_word_long_prob_log_file = None
    adaptive_word_long_prob_writer = None
    adaptive_word_summary_prob_log_file = None
    adaptive_word_summary_prob_writer = None
    adaptive_prob_window = torch.zeros(6, dtype=torch.float64)
    if args.span_masking in ("adaptive_wwm_token", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word") or uses_adaptive_word_stats:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        adaptive_prob_log_file = open(os.path.join(args.output_path, "adaptive_wwm_token_probs.csv"), "w")
        adaptive_prob_log_file.write(
            "step,epoch,avg_p_wwm,avg_p_token,chosen_wwm_frac,chosen_token_frac,"
            "masked_tokens,nonpad_tokens,actual_mask_ratio\n"
        )
    if uses_adaptive_word_stats:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        if vocab_list is None:
            vocab_list = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
        adaptive_word_long_prob_log_file = open(
            os.path.join(args.output_path, "adaptive_word_mode_probs.csv"), "w", newline=""
        )
        adaptive_word_long_prob_writer = csv.writer(adaptive_word_long_prob_log_file)
        adaptive_word_long_prob_writer.writerow([
            "step", "word_id", "word", "wwm_prob", "token_prob",
            "start_prob", "inside_prob", "wwm_gt_token",
            "wwm_gt_start", "wwm_gt_inside", "max_mode",
            "wwm_gt_token_frac", "token_gt_wwm_frac",
            "wwm_gt_start_frac", "wwm_gt_inside_frac", "wwm_is_max_frac",
            "start_is_max_frac", "inside_is_max_frac",
        ])

    # Pre-apply seq_len switches that the resumed run has already reached.
    #
    # A seq_len switch fires at the *start* of its scheduled epoch. Therefore,
    # while walking epochs before/at resume_step, apply the switch before
    # computing that epoch's dataloader length. This matters when resume_step is
    # inside or after the first epoch with the new sequence length.
    #
    # epoch_steps_record[i] = step count for completed epoch i. The main loop
    # uses these to skip completed epochs without relying on the current
    # dataloader length.
    epoch_steps_record = []
    if resume_step > 0:
        temp_step = 0
        for epoch in range(args.epochs):
            while len(args.max_seq_len) > 0 and epoch >= args.max_seq_len[0][0]:
                _, switch_seq_len = args.max_seq_len[0]
                train_dataloader, eval_dataloader = regroup_texts(args, switch_seq_len)
                args.max_seq_len = args.max_seq_len[1:]

            epoch_dl_len = len(train_dataloader)
            epoch_end = temp_step + epoch_dl_len
            if epoch_end <= resume_step:
                epoch_steps_record.append(epoch_dl_len)
                temp_step = epoch_end
            else:
                # Resume point is within this epoch — stop
                break
    print(f"[Resume] seq_len after resume pre-processing: {args.cur_max_seq_len}, remaining switches: {args.max_seq_len}", flush=True)
    print(f"[Resume] skipped epochs: {len(epoch_steps_record)}, epoch_steps_record: {epoch_steps_record}, resume_step: {resume_step}", flush=True)

    with tqdm(total=args.total_steps, initial=resume_step) as pbar:
        for epoch in range(args.epochs):
            # Delayed freeze: apply at freeze_from_epoch
            if args.freeze_from_epoch >= 0 and epoch == args.freeze_from_epoch:
                print(f"[Freeze] Epoch {epoch}: applying delayed freeze.", flush=True)
                apply_freeze(model, args)

            # In-place span_geo_p curriculum: same span_geo masking type throughout,
            # only the geometric parameter changes over training.
            if args.span_geo_p_ramp_target is not None:
                # Stepwise linear ramp: span_geo_p increases each epoch from
                # _span_geo_p_early (epoch 0) to span_geo_p_ramp_target (final epoch).
                if args.epochs > 1:
                    frac = epoch / (args.epochs - 1)
                else:
                    frac = 1.0
                args.span_geo_p = args._span_geo_p_early + frac * (args.span_geo_p_ramp_target - args._span_geo_p_early)
                print(f"[Curriculum] Epoch {epoch}: span_geo_p ramped to {args.span_geo_p:.4f}", flush=True)
            elif args.span_geo_p_late is not None:
                if epoch < args.span_geo_p_switch_epoch:
                    args.span_geo_p = args._span_geo_p_early
                else:
                    if epoch == args.span_geo_p_switch_epoch:
                        print(f"[Curriculum] Epoch {epoch}: span_geo_p switching from {args._span_geo_p_early} to {args.span_geo_p_late}", flush=True)
                    args.span_geo_p = args.span_geo_p_late

            # Forward curriculum: token first → span later
            if args.token_epochs > 0:
                if epoch < args.token_epochs:
                    args._active_span_masking = "token"
                else:
                    if epoch == args.token_epochs:
                        print(f"[Curriculum] Epoch {epoch}: switching from 'token' to '{args.span_masking}'", flush=True)
                        if args.curriculum_lr_reset:
                            remaining_steps = args.total_steps - global_step
                            new_lr = args.lr * 0.7
                            for pg in optimizer.param_groups:
                                pg['lr'] = new_lr
                            scheduler = get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=remaining_steps // 20,
                                num_training_steps=remaining_steps,
                            )
                            print(f"[Curriculum] Reset scheduler: remaining={remaining_steps}, warmup={remaining_steps//20}, new_peak_lr={new_lr:.2e}", flush=True)
                    args._active_span_masking = args.span_masking
            # Reverse curriculum: span first → token later
            elif args.span_epochs > 0:
                if epoch < args.span_epochs:
                    args._active_span_masking = args.span_masking
                elif args.transition_epochs > 0 and args.transition_masking and epoch < args.span_epochs + args.transition_epochs:
                    if epoch == args.span_epochs:
                        print(
                            f"[Curriculum] Epoch {epoch}: switching from '{args.span_masking}' to transition '{args.transition_masking}'",
                            flush=True,
                        )
                    args._active_span_masking = args.transition_masking
                else:
                    switch_epoch = args.span_epochs + args.transition_epochs if args.transition_epochs > 0 and args.transition_masking else args.span_epochs
                    if epoch == switch_epoch:
                        prev_mode = args.transition_masking if args.transition_epochs > 0 and args.transition_masking else args.span_masking
                        print(f"[Curriculum] Epoch {epoch}: switching from '{prev_mode}' to '{args.post_span_masking}'", flush=True)
                        if args.curriculum_lr_reset:
                            remaining_steps = args.total_steps - global_step
                            new_lr = args.lr * 0.7
                            for pg in optimizer.param_groups:
                                pg['lr'] = new_lr
                            scheduler = get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=remaining_steps // 20,
                                num_training_steps=remaining_steps,
                            )
                            print(f"[Curriculum] Reset scheduler: remaining={remaining_steps}, warmup={remaining_steps//20}, new_peak_lr={new_lr:.2e}", flush=True)
                    args._active_span_masking = args.post_span_masking
            # Loss-plateau curriculum: WWM first, token once WWM no longer improves enough.
            elif args.span_masking == "wwm_token_plateau":
                args._active_span_masking = "token" if plateau_switch_state["switched"] else "wwm"
            # No curriculum
            else:
                args._active_span_masking = args.span_masking

            update_wwm_mix_probs(args, epoch)
            if args._active_span_masking == "wwm_mix":
                print(
                    f"[WWM mix] Epoch {epoch}: whole={args._active_wwm_mix_whole_prob:.3f}, "
                    f"inside={args._active_wwm_mix_inside_prob:.3f}, "
                    f"start={args._active_wwm_mix_start_prob:.3f}",
                    flush=True,
                )
            elif args._active_span_masking == "wwm_token_mix":
                print(
                    f"[WWM/token mix] Epoch {epoch}: wwm={args.wwm_token_mix_wwm_prob:.3f}, "
                    f"token={1.0 - args.wwm_token_mix_wwm_prob:.3f}",
                    flush=True,
                )
            elif args.span_masking == "wwm_token_plateau":
                print(
                    f"[WWM plateau switch] Epoch {epoch}: active={args._active_span_masking}",
                    flush=True,
                )
            elif args._active_span_masking == "adaptive_wwm_token":
                args._active_adaptive_wwm_prob = adaptive_wwm_token_state["wwm_prob"]
                print(
                    f"[Adaptive WWM/token] Epoch {epoch}: wwm={args._active_adaptive_wwm_prob:.3f}, "
                    f"token={1.0 - args._active_adaptive_wwm_prob:.3f}, "
                    f"epsilon={_adaptive_epsilon(args, epoch):.3f}",
                    flush=True,
                )
            elif args._active_span_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word"):
                if args._active_span_masking == "adaptive_wwm_token_word":
                    print(
                        f"[Word adaptive WWM/token] Epoch {epoch}: "
                        f"initial/default_wwm={args.adaptive_wwm_token_initial_wwm_prob:.3f}, "
                        f"epsilon={_adaptive_epsilon(args, epoch):.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Word adaptive WWM/start/inside] Epoch {epoch}: "
                        f"default={args.adaptive_wwm_token_initial_wwm_prob:.2f}/"
                        f"{args.adaptive_wwm_start_inside_initial_start_prob:.2f}/"
                        f"{args.adaptive_wwm_start_inside_initial_inside_prob:.2f}, "
                        f"epsilon={_adaptive_epsilon(args, epoch):.3f}",
                        flush=True,
                    )

            # Skip epochs fully completed before resume_step (pre-processing already applied
            # their seq_len switches). Do NOT apply seq_len switch for skipped epochs.
            if epoch < len(epoch_steps_record):
                continue

            # Apply seq_len switch for this epoch (not skipped, not yet applied).
            if len(args.max_seq_len) > 0 and epoch >= args.max_seq_len[0][0]:
                train_dataloader, eval_dataloader = regroup_texts(args, args.max_seq_len[0][1])
                args.max_seq_len = args.max_seq_len[1:]
                print(f"[SeqLen] Epoch {epoch}: switched to seq_len={args.cur_max_seq_len}, batch_size={args.batch_size}, dl_len={len(train_dataloader)}", flush=True)

            # For the first non-skipped epoch (the resume epoch), skip steps already trained.
            # abs_step = sum of all recorded epoch steps + current step within this epoch.
            steps_before_this_epoch = sum(epoch_steps_record)
            is_resume_epoch = (resume_step > 0 and epoch == len(epoch_steps_record))
            print(
                f"[Epoch {epoch} start] global_step={global_step}, resume_step={resume_step}, "
                f"is_resume_epoch={is_resume_epoch}, abs_step_at_epoch_start={steps_before_this_epoch}, "
                f"seq_len={args.cur_max_seq_len}, batch_size={args.batch_size}, "
                f"dl_len={len(train_dataloader)}, masking={args._active_span_masking}",
                flush=True,
            )

            for step, batch in enumerate(train_dataloader):
                # Skip steps in the resume epoch that were already trained before the checkpoint.
                if is_resume_epoch and steps_before_this_epoch + step < resume_step:
                    continue
                masked_batch = mask_batch(batch,
                                          tokenizer, None if args.regular_mlm else mask_weights.to(device="cpu"),
                                          mlm_prob=args.mlm_prob,
                                          mask_replace_prob=args.mask_replace_prob,
                                          random_replace_prob=args.random_replace_prob,
                                          span_masking=args._active_span_masking,
                                          span_len=args.span_len,
                                          span_geo_p=args.span_geo_p,
                                          span_stats_mode=args.span_stats_mode,
                                          morph_inside_weight=args.morph_inside_weight,
                                          morph_start_weight=args.morph_start_weight,
                                          wwm_token_mix_wwm_prob=args.wwm_token_mix_wwm_prob,
                                          adaptive_wwm_token_wwm_prob=args._active_adaptive_wwm_prob,
                                          adaptive_word_stats=adaptive_word_stats,
                                          adaptive_epoch=epoch,
                                          adaptive_args=args,
                                          wwm_mix_whole_prob=args._active_wwm_mix_whole_prob,
                                          wwm_mix_inside_prob=args._active_wwm_mix_inside_prob,
                                          wwm_mix_start_prob=args._active_wwm_mix_start_prob
                                          )

                if adaptive_prob_log_file is not None and args._active_span_masking in ("adaptive_wwm_token", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word"):
                    adaptive_prob_window += masked_batch["adaptive_stats"].sum(dim=0).double()

                # split batch into grad_acc chunks
                batches = split_batch(masked_batch, args)
                
                for minibatch in batches:

                    with torch.autocast(dtype=AMP_DTYPE, device_type="cuda:0"):
                        if args.flops:
                            model.eval()
                            flops_inputs = {k: v for k, v in move_dict_to_cuda(minibatch).items()
                                            if k not in ("original_ids", "span_ids", "span_boundaries", "mask_mode_ids", "mask_mode_pos_ids", "word_key_ids", "adaptive_stats")}
                            flops = FlopCountAnalysis(model, tuple(flops_inputs.values()))
                            flops = flops.by_operator()
                            total = 0
                            for key in flops:
                                total += flops[key]
                            print(f"Estimated Total FLOPs: {total*3*args.total_steps}")
                            exit()


                        model_inputs = {k: v for k, v in move_dict_to_cuda(minibatch).items()
                                        if k not in ("original_ids", "span_ids", "span_boundaries", "mask_mode_ids", "mask_mode_pos_ids", "word_key_ids", "adaptive_stats", "phrase_ids")}
                        use_sbo = sbo_head is not None and args._active_span_masking not in ("token", "morph_token", "wwm_token_mix", "adaptive_wwm_token", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word", "wwm_mix")
                        use_wwm_pos = wwm_pos_embeddings is not None and args._active_span_masking in args._wwm_pos_modes
                        outputs = model(**model_inputs, output_hidden_states=(use_sbo or use_wwm_pos))
                        loss = outputs.loss

                        # SBO loss
                        if use_sbo and "span_boundaries" in minibatch:
                            hidden_states = outputs.hidden_states[-1]  # 最后一层 (batch, seq, hidden)
                            sbo_loss = compute_sbo_loss(
                                hidden_states,
                                minibatch["labels"].to(device="cuda:0"),
                                minibatch["span_boundaries"].to(device="cuda:0"),
                                sbo_head,
                                sbo_position_embeddings.weight,
                            )
                            loss = loss + sbo_loss

                        # Lightweight word-internal position auxiliary loss for WWM-like modes.
                        # It keeps the saved checkpoint architecture unchanged.
                        if use_wwm_pos:
                            hidden_states = outputs.hidden_states[-1]
                            wwm_pos_loss = compute_wwm_pos_loss(
                                hidden_states,
                                minibatch["labels"].to(device="cuda:0"),
                                minibatch["original_ids"].to(device="cuda:0"),
                                model.cls,
                                wwm_pos_embeddings,
                                tokenizer.vocab_is_word_start,
                                tokenizer.vocab_is_special,
                                args.wwm_pos_scheme,
                            )
                            loss = loss + args.wwm_pos_weight * wwm_pos_loss

                    if adaptive_wwm_token_state is not None and args._active_span_masking == "adaptive_wwm_token":
                        adaptive_wwm_token_state = update_adaptive_wwm_token_state(
                            adaptive_wwm_token_state,
                            outputs.logits.detach(),
                            minibatch["labels"].to(device="cuda:0"),
                            minibatch["mask_mode_ids"].to(device="cuda:0"),
                            args,
                            epoch,
                        )
                        args._active_adaptive_wwm_prob = adaptive_wwm_token_state["wwm_prob"]

                    if (
                        adaptive_word_stats is not None
                        and args._active_span_masking in ("wwm", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word")
                        and global_step % max(args.adaptive_word_stats_update_steps, 1) == 0
                    ):
                        update_adaptive_word_stats(
                            adaptive_word_stats,
                            outputs.logits.detach(),
                            minibatch["labels"].to(device="cuda:0"),
                            minibatch["span_ids"],
                            minibatch["mask_mode_pos_ids"],
                            minibatch["word_key_ids"],
                            args,
                        )

                    # Update stats per-minibatch and discard logits
                    if not args.regular_mlm:
                        use_span = args._active_span_masking in (
                            "wwm", "wwm_mix", "wwm_span_fixed", "wwm_span_geo",
                            "span_fixed", "span_geo", "span"
                        )
                        use_exact = use_span and args.span_stats_mode == "exact"
                        use_average = use_span and args.span_stats_mode == "average"
                        with torch.no_grad():
                            if args.soft:
                                if use_exact:
                                    mask_stats = get_batch_accuracy_soft_exact(
                                        minibatch["input_ids"].to(device="cuda:0"),
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        mask_stats,
                                    )
                                elif use_average:
                                    mask_stats = get_batch_accuracy_soft_average(
                                        minibatch["input_ids"].to(device="cuda:0"),
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        minibatch["span_ids"].to(device="cuda:0"),
                                        mask_stats,
                                    )
                                else:
                                    # token-level masking: original behaviour (only [MASK] positions)
                                    mask_stats = get_batch_accuracy_soft(
                                        minibatch["input_ids"].to(device="cuda:0"),
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        mask_stats,
                                    )
                            else:
                                if use_exact:
                                    mask_stats = get_batch_accuracy_exact(
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        mask_stats,
                                    )
                                elif use_average:
                                    mask_stats = get_batch_accuracy_average(
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        minibatch["span_ids"].to(device="cuda:0"),
                                        mask_stats,
                                    )
                                else:
                                    # token-level masking
                                    mask_stats = get_batch_accuracy(
                                        outputs.logits.detach(),
                                        minibatch["labels"].to(device="cuda:0"),
                                        mask_stats,
                                    )

                    loss = loss / args.grad_acc # To ensure consistent gradient magnitude
                    loss.backward()

                clip_params = all_params if (sbo_head is not None or wwm_pos_embeddings is not None) else list(model.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # stats already updated per-minibatch; nothing to do here

                if plateau_switch_state is not None and not plateau_switch_state["switched"]:
                    plateau_switch_state["loss_sum"] += float(loss.item() * args.grad_acc)
                    plateau_switch_state["loss_count"] += 1

                    check_steps = args.plateau_switch_check_steps or args.mask_update_steps
                    epoch_float = global_step * args.epochs / args.total_steps
                    force_due_to_epoch = (
                        args.plateau_switch_max_epochs > 0
                        and epoch_float >= args.plateau_switch_max_epochs
                    )
                    if (
                        global_step != 0
                        and global_step >= args.first_mask_update
                        and (global_step % check_steps == 0 or force_due_to_epoch)
                    ):
                        if maybe_update_plateau_switch(plateau_switch_state, args, global_step, epoch_float):
                            args._active_span_masking = "token"
                            if args.curriculum_lr_reset:
                                remaining_steps = args.total_steps - global_step
                                new_lr = args.lr * 0.7
                                for pg in optimizer.param_groups:
                                    pg['lr'] = new_lr
                                scheduler = get_cosine_schedule_with_warmup(
                                    optimizer,
                                    num_warmup_steps=remaining_steps // 20,
                                    num_training_steps=remaining_steps,
                                )
                                print(
                                    f"[WWM plateau switch] Reset scheduler: remaining={remaining_steps}, "
                                    f"warmup={remaining_steps//20}, new_peak_lr={new_lr:.2e}",
                                    flush=True,
                                )
                            print(
                                f"[WWM plateau switch] step={global_step}, epoch={epoch_float:.2f}: "
                                "switching from 'wwm' to 'token'",
                                flush=True,
                            )

                if (
                    global_step % args.mask_update_steps == 0
                    and global_step != 0
                    and global_step >= args.first_mask_update
                    and not args.regular_mlm
                ):
                    if args.soft:
                        mask_weights = update_mask_weights_soft(mask_weights, mask_stats, args.mlm_prob)
                    else:
                        mask_weights = update_mask_weights(mask_weights, mask_stats, args.mlm_prob)

                    if args.log_mlm_probs:
                        # convert probs to .4f string
                        probs_str = [f"{prob.item():.4f}" for prob in mask_weights]
                        probs_str = ",".join(probs_str)
                        log_file.write(f"{global_step},{probs_str}\n")
                        log_file.flush()
                        

                    # print top 5 tokens and their probs
                    top_tokens = torch.topk(mask_weights, 5)
                    print("Top tokens: ", flush=True, end="")
                    print_str = []
                    for token, prob in zip(tokenizer.convert_ids_to_tokens(top_tokens.indices), top_tokens.values):
                        print_str.append(f"{token} ({prob.item():.2f})")

                    print("     ".join(print_str), flush=True)

                    # print bottom 5 tokens and their probs
                    bottom_tokens = torch.topk(-mask_weights, 5)
                    print("Bottom tokens: ", flush=True, end="")
                    print_str = []
                    for token, prob in zip(tokenizer.convert_ids_to_tokens(bottom_tokens.indices), -bottom_tokens.values):
                        print_str.append(f"{token} ({prob.item():.2f})")

                    print("     ".join(print_str), flush=True)

                    mask_stats = reset_stats(mask_stats)

                if (
                    adaptive_word_stats is not None
                    and adaptive_word_long_prob_log_file is not None
                    and global_step % args.mask_update_steps == 0
                    and global_step != 0
                    and global_step >= args.first_mask_update
                ):
                    if args.span_masking == "adaptive_wwm_start_inside_word" or args.transition_masking == "adaptive_wwm_start_inside_word":
                        adaptive_word_wwm_probs, adaptive_word_start_probs, adaptive_word_inside_probs = (
                            adaptive_word_wwm_start_inside_probabilities(adaptive_word_stats, args, epoch)
                        )
                        adaptive_word_token_probs = adaptive_word_start_probs + adaptive_word_inside_probs
                        start_probs_for_long = adaptive_word_start_probs
                        inside_probs_for_long = adaptive_word_inside_probs
                    else:
                        adaptive_word_wwm_probs = adaptive_word_wwm_probabilities(
                            adaptive_word_stats, args, epoch
                        )
                        adaptive_word_token_probs = 1.0 - adaptive_word_wwm_probs
                        start_probs_for_long = None
                        inside_probs_for_long = None
                    if adaptive_word_long_prob_writer is not None:
                        wwm_gt_token = adaptive_word_wwm_probs > adaptive_word_token_probs
                        token_gt_wwm = adaptive_word_token_probs > adaptive_word_wwm_probs
                        wwm_gt_token_frac = wwm_gt_token.float().mean().item()
                        token_gt_wwm_frac = token_gt_wwm.float().mean().item()
                        if start_probs_for_long is not None and inside_probs_for_long is not None:
                            wwm_gt_start = adaptive_word_wwm_probs > start_probs_for_long
                            wwm_gt_inside = adaptive_word_wwm_probs > inside_probs_for_long
                            mode_probs = torch.stack(
                                [adaptive_word_wwm_probs, start_probs_for_long, inside_probs_for_long], dim=1
                            )
                            max_modes = torch.argmax(mode_probs, dim=1)
                            max_mode_names = ("wwm", "start", "inside")
                            wwm_gt_start_frac = wwm_gt_start.float().mean().item()
                            wwm_gt_inside_frac = wwm_gt_inside.float().mean().item()
                            wwm_is_max_frac = (max_modes == 0).float().mean().item()
                            start_is_max_frac = (max_modes == 1).float().mean().item()
                            inside_is_max_frac = (max_modes == 2).float().mean().item()
                            adaptive_word_long_prob_writer.writerows(
                                (
                                    global_step,
                                    word_id,
                                    vocab_list[word_id],
                                    f"{adaptive_word_wwm_probs[word_id].item():.6f}",
                                    f"{adaptive_word_token_probs[word_id].item():.6f}",
                                    f"{start_probs_for_long[word_id].item():.6f}",
                                    f"{inside_probs_for_long[word_id].item():.6f}",
                                    int(wwm_gt_token[word_id].item()),
                                    int(wwm_gt_start[word_id].item()),
                                    int(wwm_gt_inside[word_id].item()),
                                    max_mode_names[int(max_modes[word_id].item())],
                                    f"{wwm_gt_token_frac:.6f}",
                                    f"{token_gt_wwm_frac:.6f}",
                                    f"{wwm_gt_start_frac:.6f}",
                                    f"{wwm_gt_inside_frac:.6f}",
                                    f"{wwm_is_max_frac:.6f}",
                                    f"{start_is_max_frac:.6f}",
                                    f"{inside_is_max_frac:.6f}",
                                )
                                for word_id in range(len(vocab_list))
                            )
                        else:
                            adaptive_word_long_prob_writer.writerows(
                                (
                                    global_step,
                                    word_id,
                                    vocab_list[word_id],
                                    f"{adaptive_word_wwm_probs[word_id].item():.6f}",
                                    f"{adaptive_word_token_probs[word_id].item():.6f}",
                                    "",
                                    "",
                                    int(wwm_gt_token[word_id].item()),
                                    "",
                                    "",
                                    "wwm" if wwm_gt_token[word_id].item() else "token",
                                    f"{wwm_gt_token_frac:.6f}",
                                    f"{token_gt_wwm_frac:.6f}",
                                    "",
                                    "",
                                    f"{wwm_gt_token_frac:.6f}",
                                    "",
                                    "",
                                )
                                for word_id in range(len(vocab_list))
                            )
                        adaptive_word_long_prob_log_file.flush()

                # ----- LOGGING -----
                if is_step("logging", global_step, args):
                    epoch_float = global_step * args.epochs / args.total_steps
                    print(f"Epoch {epoch_float:.2f}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

                    if args.wandb:
                        wandb.log({
                            "epoch": epoch_float,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0]
                        })
                    if adaptive_wwm_token_state is not None and args._active_span_masking == "adaptive_wwm_token":
                        print(
                            "[Adaptive WWM/token] "
                            f"p_wwm={adaptive_wwm_token_state['wwm_prob']:.3f}, "
                            f"wwm_loss={adaptive_wwm_token_state['wwm_loss_ema']:.4f}, "
                            f"token_loss={adaptive_wwm_token_state['token_loss_ema']:.4f}, "
                            f"wwm_imp={adaptive_wwm_token_state['wwm_improvement']:.4f}, "
                            f"token_imp={adaptive_wwm_token_state['token_improvement']:.4f}",
                            flush=True,
                        )
                    if adaptive_word_stats is not None and args._active_span_masking in ("adaptive_wwm_token_word", "adaptive_wwm_start_inside_word"):
                        summary = summarize_adaptive_word_stats(adaptive_word_stats)
                        if args._active_span_masking == "adaptive_wwm_start_inside_word":
                            print(
                                "[Word adaptive WWM/start/inside] "
                                f"wwm_seen={summary['wwm_seen']}, "
                                f"start_seen={summary['start_seen']}, "
                                f"inside_seen={summary['inside_seen']}, "
                                f"three_seen={summary['three_seen']}, "
                                f"wwm_loss={summary['wwm_loss']:.4f}, "
                                f"start_loss={summary['start_loss']:.4f}, "
                                f"inside_loss={summary['inside_loss']:.4f}, "
                                f"wwm_imp={summary['wwm_imp']:.4f}, "
                                f"start_imp={summary['start_imp']:.4f}, "
                                f"inside_imp={summary['inside_imp']:.4f}",
                                flush=True,
                            )
                        else:
                            print(
                                "[Word adaptive WWM/token] "
                                f"wwm_seen={summary['wwm_seen']}, "
                                f"token_seen={summary['token_seen']}, "
                                f"both_seen={summary['both_seen']}, "
                                f"wwm_loss={summary['wwm_loss']:.4f}, "
                                f"token_loss={summary['token_loss']:.4f}, "
                                f"wwm_imp={summary['wwm_imp']:.4f}, "
                                f"token_imp={summary['token_imp']:.4f}",
                                flush=True,
                            )
                    if adaptive_prob_log_file is not None and args._active_span_masking in ("adaptive_wwm_token", "adaptive_wwm_token_word", "adaptive_wwm_start_inside_word"):
                        prob_count = adaptive_prob_window[1].item()
                        chosen_count = (adaptive_prob_window[2] + adaptive_prob_window[3]).item()
                        nonpad_tokens = adaptive_prob_window[5].item()
                        avg_p_wwm = adaptive_prob_window[0].item() / prob_count if prob_count > 0 else 0.0
                        chosen_wwm_frac = adaptive_prob_window[2].item() / chosen_count if chosen_count > 0 else 0.0
                        masked_tokens = adaptive_prob_window[4].item()
                        actual_mask_ratio = masked_tokens / nonpad_tokens if nonpad_tokens > 0 else 0.0
                        adaptive_prob_log_file.write(
                            f"{global_step},{epoch_float:.4f},{avg_p_wwm:.6f},{1.0 - avg_p_wwm:.6f},"
                            f"{chosen_wwm_frac:.6f},{1.0 - chosen_wwm_frac:.6f},"
                            f"{masked_tokens:.0f},{nonpad_tokens:.0f},{actual_mask_ratio:.6f}\n"
                        )
                        adaptive_prob_log_file.flush()
                        adaptive_prob_window.zero_()

                # ----- EVALUATION -----
                if is_step("eval", global_step, args):
                    metrics = evaluate(model, tokenizer, eval_dataloader, args)
                    print(f"----- Eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f} -----", flush=True)

                    if args.wandb:
                        wandb.log({
                            "eval_acc": metrics["acc"],
                            "eval_loss": metrics["loss"]
                        })

                # ----- SAVING -----
                if is_step("save", global_step, args):
                    save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"----- Saved checkpoint to: {save_path} -----", flush=True)

                pbar.update(1)
                global_step += 1
                if args.mask_decay > 0:
                    args.mlm_prob = args.mlm_prob - (args.mask_decay/args.total_steps)

    metrics = evaluate(model, tokenizer, eval_dataloader, args)
    print(f"Final eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f}", flush=True)

    save_path = os.path.join(args.output_path, f"checkpoint-{args.total_steps}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    if args.wandb:
        wandb.finish()
    if log_file is not None:
        log_file.close()
    if adaptive_prob_log_file is not None:
        adaptive_prob_log_file.close()
    if adaptive_word_wwm_prob_log_file is not None:
        adaptive_word_wwm_prob_log_file.close()
    if adaptive_word_token_prob_log_file is not None:
        adaptive_word_token_prob_log_file.close()
    if adaptive_word_start_prob_log_file is not None:
        adaptive_word_start_prob_log_file.close()
    if adaptive_word_inside_prob_log_file is not None:
        adaptive_word_inside_prob_log_file.close()
    if adaptive_word_long_prob_log_file is not None:
        adaptive_word_long_prob_log_file.close()
    if adaptive_word_summary_prob_log_file is not None:
        adaptive_word_summary_prob_log_file.close()

def parse_max_seq_len(max_seq_len):
    if "," in max_seq_len:
        max_seq_len = max_seq_len.split(",")
        assert ":" in max_seq_len[0]
        return [(int(val.split(":")[0]), int(val.split(":")[1])) for val in max_seq_len]
    elif ":" in max_seq_len:
        max_seq_len = max_seq_len.split(":")[1]
    return [(0, int(max_seq_len))]

def main():
    args = parser.parse_args()
    args.max_seq_len = parse_max_seq_len(args.max_seq_len)
    args._span_geo_p_early = args.span_geo_p
    set_seed(args.seed)

    if args.wandb:
        assert wandb_available
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(
            project='babylm25',
            name=output_dir,
            config=vars(args),   
        )

    tokenizer = DebertaV2Tokenizer(args.tokenizer, do_lower_case=args.lower)

    # 预计算词表级别的词首标记（▁开头）和特殊token集合，供 WWM 使用
    _special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                    tokenizer.mask_token_id, tokenizer.pad_token_id,
                    tokenizer.bos_token_id, tokenizer.eos_token_id}
    _special_ids.discard(None)
    _all_tokens = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    tokenizer.vocab_is_word_start = torch.tensor(
        [bool(s and s.startswith('▁')) for s in _all_tokens], dtype=torch.bool
    )
    tokenizer.vocab_is_alpha = torch.tensor(
        [bool(s and s.lstrip('▁').isalpha()) for s in _all_tokens], dtype=torch.bool
    )
    tokenizer.vocab_is_special = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
    for sid in _special_ids:
        if sid is not None and sid < tokenizer.vocab_size:
            tokenizer.vocab_is_special[sid] = True

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = 1024
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.sep_token_id = tokenizer.sep_token_id


    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.dropout = args.dropout
    config.hidden_dropout_prob = args.dropout

    if args.custom == "nhot":
        from ngram_model import NGramDebertaV2ForMaskedLM
        model = NGramDebertaV2ForMaskedLM(config, tokenizer)
    elif args.custom == "lora":
        from ngram_model import LoraModelForMaskedLM
        model = LoraModelForMaskedLM(config)
    else:
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    if args.pretrained:
        model = model.from_pretrained(args.model_path)

    if args.resume_from_checkpoint is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.resume_from_checkpoint, trust_remote_code=True)
        print(f"Loaded model weights from {args.resume_from_checkpoint}", flush=True)

    # Freeze encoder layers if requested (immediate, not delayed)
    if args.freeze_from_epoch == -1:
        apply_freeze(model, args)

    # print model num parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")

    # for p in model.named_parameters():
    #     print(p[0], p[1].shape)

    args._use_phrase_grouping = (args.span_masking == "phrase" and bool(args.phrase_data))

    if args._use_phrase_grouping:
        # Pre-parsed dataset already has input_ids + phrase_ids columns
        from datasets import DatasetDict
        train_ds = load_from_disk(args.phrase_data)
        val_ds = load_dataset('text', data_files=args.valid_data)['train']
        val_ds = val_ds.map(tokenize,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer, 'input_field': 'text'},
            remove_columns=val_ds.column_names,
            num_proc=args.cpus,
            desc="Tokenizing validation",
        )
        dataset = DatasetDict({"train": train_ds, "validation": val_ds})
    elif args.custom == "pos":
        dataset = load_from_disk(args.train_data)
        dataset['validation'] = load_dataset('text', data_files=args.valid_data)['train']
    else:
        dataset = load_dataset('text', data_files={'train': args.train_data, 'validation': args.valid_data})

    if args.debug:
        dataset['train'] = dataset['train'].select(range(100))
        dataset['validation'] = dataset['validation'].select(range(100))

    if not args._use_phrase_grouping:
        dataset = dataset.map(tokenize,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer, 'input_field': 'text'},
            remove_columns=dataset["train"].column_names,
            num_proc=args.cpus,
            desc="Tokenizing",
            # load_from_cache_file=False,
        )
        
    args.dataset = dataset

    print(dataset)

    max_seq_len = args.max_seq_len.pop(0)[1]
    args.init_max_seq_len = max_seq_len
    args.cur_max_seq_len = max_seq_len
    args.dataset_len_lines = len(dataset['train'])
    args.tokens_per_1000 = dataset['train'].map(calculate_num_tokens, 
                                                       batched = True,
                                                       num_proc=args.cpus, 
                                                       remove_columns=dataset["train"].column_names)['num_tokens']

    args.dataset_len_tokens = sum(args.tokens_per_1000)
    args.total_steps = calculate_total_steps(args)

    args.is_strict_small = (args.dataset_len_tokens // 10e6) < 10 # assuming there are fewer than 10 tokens per word

    if args.is_strict_small:
        steps_1m = np.round(np.linspace(args.total_steps//100, args.total_steps//10, 10)).astype(int)
        steps_10m = np.round(np.linspace(args.total_steps//10, args.total_steps, 10)).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:]
    else: # strict
        steps_1m = np.linspace(args.total_steps//1000, args.total_steps//100, 10).astype(int)
        steps_10m = np.linspace(args.total_steps//100, args.total_steps//10, 10).astype(int)
        steps_100m = np.linspace(args.total_steps//10, args.total_steps, 10).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:] + list(steps_100m)[1:]


    print(f"Dataset length: {args.dataset_len_lines} lines, {args.dataset_len_tokens} tokens", flush=True)
    print(f"Total steps: {args.total_steps}", flush=True)
    print(f"Save checkpoints: {args.checkpoints}", flush=True)


    if args._use_phrase_grouping:
        from datasets import DatasetDict as _DatasetDict
        grouped_dataset = _DatasetDict({
            "train": dataset['train'].map(group_texts_with_phrases,
                batched=True,
                fn_kwargs={'max_len': max_seq_len},
                num_proc=1,
                desc="Grouping train",
            ),
            "validation": dataset['validation'].map(group_texts,
                batched=True,
                fn_kwargs={'max_len': max_seq_len},
                num_proc=args.cpus,
                desc="Grouping validation",
            ),
        })
    else:
        grouped_dataset = dataset.map(group_texts,
            batched=True,
            fn_kwargs={'max_len': max_seq_len},
            num_proc=args.cpus,
            desc="Grouping",
        )

    print("Dataset length after grouping:", len(grouped_dataset['train']))

    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['train'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=True, 
        collate_fn=padding_collate_fn
        )
    
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset['validation'], 
        batch_size=args.batch_size, 
        num_workers=args.cpus,
        shuffle=False, 
        collate_fn=padding_collate_fn
        )

    train(args, model, tokenizer, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()
