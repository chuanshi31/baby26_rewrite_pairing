# train_mask.py clean cut: token + whole-word masking only.
# This file keeps the regular-MLM token/WWM path from train_mask.py and removes
# AMLM, adaptive masking, non-WWM span modes, phrase/morph modes, SBO and aux losses.

import argparse
import os
import math
import re as _re
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from transformers import AutoConfig, AutoModelForMaskedLM, DebertaV2Tokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset

from preprocessing import tokenize, padding_collate_fn, group_texts
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
parser.add_argument("--all_checkpoints", action="store_true", help="Save and evaluate model every 1/10/100M tokens, per challenge stipulations. Overrides eval_steps and save_steps.")
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--intermediate_size", type=int, default=3072)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--mlm_prob", type=float, default=0.15)
parser.add_argument("--mask_replace_prob", type=float, default=0.8)
parser.add_argument("--random_replace_prob", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--wandb", action="store_true", help="Report to wandb")
parser.add_argument("--lamb", action="store_true", help="LAMB optimization")
parser.add_argument("--lower", action="store_true", help="Lowercase")
parser.add_argument("--mask_decay", type=float, default=0.0, help="Mask decay. e.g. 0.1 means decay by 0.1 over the course of training")
parser.add_argument("--span_masking", type=str, default="token", choices=["token", "wwm"], help="Masking strategy: token or wwm")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from. Resumes at the step encoded in the directory name.")
parser.add_argument("--span_epochs", type=int, default=0, help="Reverse curriculum: use span_masking for first N epochs, then switch to token masking. 0 means always use span_masking.")
parser.add_argument("--token_epochs", type=int, default=0, help="Forward curriculum: use token masking for first N epochs, then switch to span_masking. 0 means always use span_masking.")
parser.add_argument("--curriculum_lr_reset", action="store_true", help="Reset LR to 0.7x with a new cosine schedule at curriculum switch.")
parser.add_argument("--freeze_lower_layers", type=int, default=0, help="Freeze encoder layers 0..N-1 + embeddings at training start. 0 = no freeze.")
parser.add_argument("--freeze_layers", type=str, default="", help="Freeze specific encoder layer indices (comma-separated) + embeddings.")


def parse_max_seq_len(max_seq_len):
    if "," in max_seq_len:
        max_seq_len = max_seq_len.split(",")
        assert ":" in max_seq_len[0]
        return [(int(val.split(":")[0]), int(val.split(":")[1])) for val in max_seq_len]
    elif ":" in max_seq_len:
        max_seq_len = max_seq_len.split(":")[1]
    return [(0, int(max_seq_len))]


def split_batch(batch, args):
    minibatch_size = args.batch_size // args.grad_acc
    if len(batch["input_ids"]) == minibatch_size:
        return [batch]
    batches = []
    for i in range(0, len(batch["input_ids"]), minibatch_size):
        minibatch = {}
        for key in batch.keys():
            minibatch[key] = batch[key][i:i + minibatch_size]
        batches.append(minibatch)
    return batches


def move_dict_to_cuda(d):
    return {key: value.to(device="cuda:0") for key, value in d.items()}


def get_amp_dtype():
    # train_mask.py behaviour: V100 falls back to fp16; Ampere/Blackwell uses bf16.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def calculate_num_tokens(examples):
    return {"num_tokens": [sum(len(examples["input_ids"][i]) for i in range(len(examples["input_ids"])))]}


def calculate_total_steps(args):
    def calc_exs_per_epoch(tokens_per_1000, max_seq_len):
        return sum([t // max_seq_len for t in tokens_per_1000])

    exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, args.init_max_seq_len)
    batches_per_epoch = math.ceil(exs_per_epoch / args.batch_size)
    cur_epoch = 0
    if len(args.max_seq_len) == 0:
        return batches_per_epoch * args.epochs
    total_steps = 0
    batch_size = args.batch_size
    prev_seq_len = args.init_max_seq_len
    for epoch_num, seq_len in args.max_seq_len:
        total_steps = total_steps + batches_per_epoch * (epoch_num - cur_epoch)
        batch_size = max(1, int(batch_size / (seq_len / prev_seq_len)))
        exs_per_epoch = calc_exs_per_epoch(args.tokens_per_1000, seq_len)
        batches_per_epoch = math.ceil(exs_per_epoch / batch_size)
        cur_epoch = epoch_num
        prev_seq_len = seq_len
    total_steps = total_steps + batches_per_epoch * (args.epochs - cur_epoch)
    return total_steps


def is_step(step_type: str, global_step: int, args):
    if args.all_checkpoints:
        return global_step in args.checkpoints
    step_arg = getattr(args, f"{step_type}_steps")
    return global_step % step_arg == 0 and global_step != 0


def get_active_masking_mode(args, epoch):
    if args.token_epochs > 0:
        return "token" if epoch < args.token_epochs else args.span_masking
    if args.span_epochs > 0:
        return args.span_masking if epoch < args.span_epochs else "token"
    return args.span_masking


def describe_curriculum(args):
    if args.token_epochs > 0:
        return f"token for epochs 0-{args.token_epochs - 1}, then {args.span_masking}"
    if args.span_epochs > 0:
        return f"{args.span_masking} for epochs 0-{args.span_epochs - 1}, then token"
    return f"always {args.span_masking}"


# Whole-word masking, retained from train_mask.py regular-MLM semantics.
def _wwm_mask_indices(seq_len, pad_token_id, enc, enc_mask_weights, mlm_prob, tokenizer, use_weights):
    not_maskable = tokenizer.vocab_is_special[enc] | (enc == pad_token_id)
    is_word_start = tokenizer.vocab_is_word_start[enc].clone()

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

    span_map = torch.full((num_words,), -1, dtype=torch.long)
    selected_wids = word_order[:n_select]
    span_map[selected_wids] = torch.arange(n_select, dtype=torch.long)

    span_ids = torch.full((seq_len,), -1, dtype=torch.long)
    span_ids[maskable] = span_map[maskable_wids]
    selected = span_ids >= 0
    return selected, span_ids


def _apply_mask_tokens(enc, selected, tokenizer, mask_replace_prob, random_replace_prob):
    rand = torch.rand(len(enc))
    to_mask = rand < mask_replace_prob
    to_replace = (rand > mask_replace_prob) & (rand < mask_replace_prob + random_replace_prob)
    to_mask = to_mask & selected
    to_replace = to_replace & selected
    randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
    enc[to_mask] = tokenizer.mask_token_id
    enc[to_replace] = randoms[to_replace]
    return enc


def _apply_mask_groups(enc, selected, span_ids, tokenizer, mask_replace_prob, random_replace_prob):
    if not selected.any():
        return enc
    max_group = int(span_ids[selected].max().item())
    if max_group < 0:
        return enc
    group_rand = torch.rand(max_group + 1)
    group_to_mask = group_rand < mask_replace_prob
    group_to_replace = (group_rand > mask_replace_prob) & (group_rand < mask_replace_prob + random_replace_prob)
    group_ids = span_ids.clamp(min=0)
    to_mask = selected & group_to_mask[group_ids]
    to_replace = selected & group_to_replace[group_ids]
    randoms = torch.randint(0, tokenizer.vocab_size, (len(enc),))
    enc[to_mask] = tokenizer.mask_token_id
    enc[to_replace] = randoms[to_replace]
    return enc


def mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15, mask_replace_prob=0.8,
               random_replace_prob=0.1, span_masking="token"):
    """Regular train_mask.py MLM path with only token and WWM modes retained."""
    amlm = mask_weights is not None
    if mask_weights is None:
        mask_weights = torch.full((tokenizer.vocab_size,), mlm_prob, device=batch["input_ids"].device)

    masked_batch = batch.copy()
    masked_batch["original_ids"] = batch["input_ids"].clone()

    batch_size = len(batch["input_ids"])
    seq_len_0 = len(batch["input_ids"][0])
    span_ids_batch = torch.full((batch_size, seq_len_0), -1, dtype=torch.long)
    mask_mode_ids = torch.full((batch_size,), -1, dtype=torch.long)

    for i in range(batch_size):
        seq_len = len(batch["input_ids"][i])
        enc_mask_weights = mask_weights[batch["input_ids"][i]]
        enc_mask_weights[batch["input_ids"][i] == tokenizer.pad_token_id] = 0

        if span_masking == "token":
            mask_mode_ids[i] = 0
            enc_mask_weights_norm = mlm_prob * seq_len * enc_mask_weights / enc_mask_weights.sum()
            enc_mask_mask = torch.rand(seq_len) < enc_mask_weights_norm
            enc = batch["input_ids"][i].clone()
            enc = _apply_mask_tokens(enc, enc_mask_mask, tokenizer, mask_replace_prob, random_replace_prob)
        elif span_masking == "wwm":
            mask_mode_ids[i] = 1
            enc = batch["input_ids"][i].clone()
            enc_mask_weights = enc_mask_weights.float()
            w_sum = enc_mask_weights.sum()
            norm_w = enc_mask_weights / w_sum if w_sum > 0 else enc_mask_weights
            enc_mask_mask, span_ids_batch[i] = _wwm_mask_indices(
                seq_len, tokenizer.pad_token_id, enc, norm_w, mlm_prob, tokenizer, use_weights=amlm
            )
            enc = _apply_mask_groups(enc, enc_mask_mask, span_ids_batch[i], tokenizer, mask_replace_prob, random_replace_prob)
        else:
            raise ValueError(f"Unsupported masking mode in clean train_mask cut: {span_masking}")

        masked_batch["input_ids"][i] = enc
        labels = batch["labels"][i].clone()
        labels[~enc_mask_mask] = -100
        masked_batch["labels"][i] = labels

    masked_batch["span_ids"] = span_ids_batch
    masked_batch["mask_mode_ids"] = mask_mode_ids
    return masked_batch


def evaluate(model, tokenizer, dataloader, args):
    model.eval()
    correct = 0
    total = 0
    avg_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch["input_ids"]) == 0:
                continue
            masked_batch = mask_batch(batch, tokenizer, mask_weights=None, mlm_prob=0.15,
                                      mask_replace_prob=0.8, random_replace_prob=0.1,
                                      span_masking=args._active_span_masking)
            for minibatch in split_batch(masked_batch, args):
                model_inputs = {k: v for k, v in move_dict_to_cuda(minibatch).items()
                                if k not in ("original_ids", "span_ids", "mask_mode_ids")}
                with torch.autocast(dtype=args._amp_dtype, device_type="cuda:0"):
                    outputs = model(**model_inputs)
                avg_loss += outputs.loss.item()
                labels = minibatch["labels"].to(device=outputs.logits.device, dtype=torch.long)
                label_mask = labels != -100
                correct += (outputs.logits.argmax(dim=-1)[label_mask] == labels[label_mask]).sum().item()
                total += label_mask.sum().item()
    model.train()
    return {"acc": 100 * correct / total if total > 0 else 0.0,
            "loss": avg_loss / (len(dataloader) * args.grad_acc)}


def regroup_texts(args, max_seq_len):
    cur_max_seq_len = args.cur_max_seq_len
    print("CUR MAX SEQ LEN:", cur_max_seq_len)

    def map_with_fallback(dataset, fn, *, batched, fn_kwargs, num_proc, desc):
        try:
            return dataset.map(fn, batched=batched, fn_kwargs=fn_kwargs, num_proc=num_proc, desc=desc)
        except Exception as exc:
            if num_proc == 1:
                raise
            print(f"[SeqLen] {desc} with num_proc={num_proc} failed: {repr(exc)}. Retrying with num_proc=1.", flush=True)
            return dataset.map(fn, batched=batched, fn_kwargs=fn_kwargs, num_proc=1, desc=f"{desc} (num_proc=1 fallback)")

    change_ratio = max_seq_len / cur_max_seq_len
    args.batch_size = max(1, int(args.batch_size / change_ratio))
    grouped_dataset = map_with_fallback(args.dataset, group_texts, batched=True,
                                        fn_kwargs={"max_len": max_seq_len}, num_proc=args.cpus,
                                        desc="Grouping")
    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset["train"], batch_size=args.batch_size, num_workers=args.cpus,
        shuffle=True, collate_fn=padding_collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset["validation"], batch_size=args.batch_size, num_workers=args.cpus,
        shuffle=False, collate_fn=padding_collate_fn)
    args.cur_max_seq_len = max_seq_len
    return train_dataloader, eval_dataloader


def _curriculum_lr_reset(optimizer, args, global_step):
    remaining_steps = args.total_steps - global_step
    new_lr = args.lr * 0.7
    for pg in optimizer.param_groups:
        pg["lr"] = new_lr
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=remaining_steps // 20, num_training_steps=remaining_steps)
    print(f"[Curriculum] Reset scheduler: remaining={remaining_steps}, warmup={remaining_steps//20}, new_peak_lr={new_lr:.2e}", flush=True)
    return scheduler


def train(args, model, tokenizer, train_dataloader, eval_dataloader):
    if args.span_epochs > 0 and args.token_epochs > 0:
        raise ValueError("Use only one curriculum direction: --span_epochs or --token_epochs, not both.")
    args._active_span_masking = get_active_masking_mode(args, 0)
    args._amp_dtype = get_amp_dtype()
    print(f"[Masking] Available modes: token, wwm", flush=True)
    print(f"[Curriculum] {describe_curriculum(args)}", flush=True)
    print(f"[AMP] dtype={args._amp_dtype}", flush=True)

    model = model.to(dtype=args._amp_dtype, device="cuda:0")
    all_params = [p for p in model.parameters() if p.requires_grad]
    if args.lamb:
        optimizer = LAMB(all_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.total_steps // 100, num_training_steps=args.total_steps)

    resume_step = 0
    if args.resume_from_checkpoint is not None:
        ckpt_name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        resume_step = int(ckpt_name.split("-")[-1])
        print(f"Resuming from checkpoint {args.resume_from_checkpoint}, step {resume_step}", flush=True)
        for _ in range(resume_step):
            scheduler.step()
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler.get_last_lr()[0]

    global_step = resume_step
    epoch_steps_record = []
    remaining_steps = resume_step
    resume_epoch = 0
    resume_step_in_epoch = 0
    for epoch in range(args.epochs):
        if args.max_seq_len and epoch >= args.max_seq_len[0][0]:
            train_dataloader, eval_dataloader = regroup_texts(args, args.max_seq_len[0][1])
            args.max_seq_len = args.max_seq_len[1:]
        dl_len = len(train_dataloader)
        if remaining_steps >= dl_len:
            epoch_steps_record.append(dl_len)
            remaining_steps -= dl_len
            resume_epoch = epoch + 1
            resume_step_in_epoch = 0
        else:
            resume_epoch = epoch
            resume_step_in_epoch = remaining_steps
            break
    else:
        resume_epoch = args.epochs
        resume_step_in_epoch = 0
    print(f"[Resume] seq_len after resume pre-processing: {args.cur_max_seq_len}, remaining switches: {args.max_seq_len}", flush=True)
    print(f"[Resume] skipped epochs: {len(epoch_steps_record)}, epoch_steps_record: {epoch_steps_record}, resume_step: {resume_step}", flush=True)

    print(f"Max seq len: {args.cur_max_seq_len}", flush=True)
    print(f"Next seq len: {args.max_seq_len}", flush=True)

    with tqdm(total=args.total_steps, initial=resume_step) as pbar:
        for epoch in range(args.epochs):
            if args.token_epochs > 0:
                if epoch < args.token_epochs:
                    args._active_span_masking = "token"
                else:
                    if epoch == args.token_epochs:
                        print(f"[Curriculum] Epoch {epoch}: switching from 'token' to '{args.span_masking}'", flush=True)
                        if args.curriculum_lr_reset:
                            scheduler = _curriculum_lr_reset(optimizer, args, global_step)
                    args._active_span_masking = args.span_masking
            elif args.span_epochs > 0:
                if epoch < args.span_epochs:
                    args._active_span_masking = args.span_masking
                else:
                    if epoch == args.span_epochs:
                        print(f"[Curriculum] Epoch {epoch}: switching from '{args.span_masking}' to 'token'", flush=True)
                        if args.curriculum_lr_reset:
                            scheduler = _curriculum_lr_reset(optimizer, args, global_step)
                    args._active_span_masking = "token"
            else:
                args._active_span_masking = args.span_masking

            if epoch < len(epoch_steps_record):
                continue

            if len(args.max_seq_len) > 0 and epoch >= args.max_seq_len[0][0]:
                train_dataloader, eval_dataloader = regroup_texts(args, args.max_seq_len[0][1])
                args.max_seq_len = args.max_seq_len[1:]
                print(f"[SeqLen] Epoch {epoch}: switched to seq_len={args.cur_max_seq_len}, batch_size={args.batch_size}, dl_len={len(train_dataloader)}", flush=True)

            steps_before_this_epoch = sum(epoch_steps_record)
            is_resume_epoch = (resume_step > 0 and epoch == len(epoch_steps_record))
            print(f"[Epoch {epoch} start] global_step={global_step}, resume_step={resume_step}, is_resume_epoch={is_resume_epoch}, abs_step_at_epoch_start={steps_before_this_epoch}, seq_len={args.cur_max_seq_len}, batch_size={args.batch_size}, dl_len={len(train_dataloader)}, masking={args._active_span_masking}", flush=True)

            for step, batch in enumerate(train_dataloader):
                if is_resume_epoch and steps_before_this_epoch + step < resume_step:
                    continue
                masked_batch = mask_batch(batch, tokenizer, mask_weights=None,
                                          mlm_prob=args.mlm_prob,
                                          mask_replace_prob=args.mask_replace_prob,
                                          random_replace_prob=args.random_replace_prob,
                                          span_masking=args._active_span_masking)
                batches = split_batch(masked_batch, args)
                for minibatch in batches:
                    model_inputs = {k: v for k, v in move_dict_to_cuda(minibatch).items()
                                    if k not in ("original_ids", "span_ids", "mask_mode_ids")}
                    with torch.autocast(dtype=args._amp_dtype, device_type="cuda:0"):
                        outputs = model(**model_inputs)
                        loss = outputs.loss
                    loss = loss / args.grad_acc
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if is_step("logging", global_step, args):
                    epoch_float = global_step * args.epochs / args.total_steps
                    print(f"Epoch {epoch_float:.2f}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    if args.wandb:
                        wandb.log({"epoch": epoch_float, "loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

                if is_step("eval", global_step, args):
                    metrics = evaluate(model, tokenizer, eval_dataloader, args)
                    print(f"----- Eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f} -----", flush=True)
                    if args.wandb:
                        wandb.log({"eval_acc": metrics["acc"], "eval_loss": metrics["loss"]})

                if is_step("save", global_step, args):
                    save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"----- Saved checkpoint to: {save_path} -----", flush=True)

                pbar.update(1)
                global_step += 1
                if args.mask_decay > 0:
                    args.mlm_prob = args.mlm_prob - (args.mask_decay / args.total_steps)

    metrics = evaluate(model, tokenizer, eval_dataloader, args)
    print(f"Final eval accuracy: {metrics['acc']:.2f}, Loss: {metrics['loss']:.4f}", flush=True)
    save_path = os.path.join(args.output_path, f"checkpoint-{args.total_steps}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if args.wandb:
        wandb.finish()


def main():
    args = parser.parse_args()
    args.max_seq_len = parse_max_seq_len(args.max_seq_len)
    set_seed(args.seed)

    if args.wandb:
        assert wandb_available
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(project="babylm25", name=output_dir, config=vars(args))

    tokenizer = DebertaV2Tokenizer(args.tokenizer, do_lower_case=args.lower)

    _special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                    tokenizer.mask_token_id, tokenizer.pad_token_id,
                    tokenizer.bos_token_id, tokenizer.eos_token_id}
    _special_ids.discard(None)
    _all_tokens = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    tokenizer.vocab_is_word_start = torch.tensor(
        [bool(s and s.startswith("▁")) for s in _all_tokens], dtype=torch.bool)
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

    model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    if args.resume_from_checkpoint is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.resume_from_checkpoint, trust_remote_code=True)
        print(f"Loaded model weights from {args.resume_from_checkpoint}", flush=True)

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
        print(f"[Freeze] Froze encoder layers {sorted(freeze_indices)} + embeddings. Frozen={frozen:,}, Trainable={trainable:,}", flush=True)
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
        print(f"[Freeze] Froze layers 0-{args.freeze_lower_layers - 1} + embeddings. Frozen={frozen:,}, Trainable={trainable:,}", flush=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")

    dataset = load_dataset("text", data_files={"train": args.train_data, "validation": args.valid_data})
    if args.debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(100))

    dataset = dataset.map(tokenize, batched=True,
                          fn_kwargs={"tokenizer": tokenizer, "input_field": "text"},
                          remove_columns=dataset["train"].column_names,
                          num_proc=args.cpus,
                          desc="Tokenizing")
    args.dataset = dataset
    print(dataset)

    max_seq_len = args.max_seq_len.pop(0)[1]
    args.init_max_seq_len = max_seq_len
    args.cur_max_seq_len = max_seq_len
    args.dataset_len_lines = len(dataset["train"])
    args.tokens_per_1000 = dataset["train"].map(
        calculate_num_tokens, batched=True, num_proc=args.cpus,
        remove_columns=dataset["train"].column_names)["num_tokens"]
    args.dataset_len_tokens = sum(args.tokens_per_1000)
    args.total_steps = calculate_total_steps(args)
    args.is_strict_small = (args.dataset_len_tokens // 10e6) < 10
    if args.is_strict_small:
        steps_1m = np.round(np.linspace(args.total_steps // 100, args.total_steps // 10, 10)).astype(int)
        steps_10m = np.round(np.linspace(args.total_steps // 10, args.total_steps, 10)).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:]
    else:
        steps_1m = np.linspace(args.total_steps // 1000, args.total_steps // 100, 10).astype(int)
        steps_10m = np.linspace(args.total_steps // 100, args.total_steps // 10, 10).astype(int)
        steps_100m = np.linspace(args.total_steps // 10, args.total_steps, 10).astype(int)
        args.checkpoints = list(steps_1m) + list(steps_10m)[1:] + list(steps_100m)[1:]

    print(f"Dataset length: {args.dataset_len_lines} lines, {args.dataset_len_tokens} tokens", flush=True)
    print(f"Total steps: {args.total_steps}", flush=True)
    print(f"Save checkpoints: {args.checkpoints}", flush=True)

    grouped_dataset = dataset.map(group_texts, batched=True,
                                  fn_kwargs={"max_len": max_seq_len},
                                  num_proc=args.cpus,
                                  desc="Grouping")
    print("Dataset length after grouping:", len(grouped_dataset["train"]))
    train_dataloader = torch.utils.data.DataLoader(
        grouped_dataset["train"], batch_size=args.batch_size, num_workers=args.cpus,
        shuffle=True, collate_fn=padding_collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(
        grouped_dataset["validation"], batch_size=args.batch_size, num_workers=args.cpus,
        shuffle=False, collate_fn=padding_collate_fn)

    train(args, model, tokenizer, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
