#!/usr/bin/env python3
"""
Paraphrase-only augmentation for fineweb_2m_new_source.txt.

For each input line (one document per line):
  - Generate 1 paraphrase (no negation).

Output: data/fineweb_2m_new_paraphrase.jsonl
        data/fineweb_2m_new_paraphrase_failed.jsonl
        data/fineweb_2m_new_paraphrase.txt

Usage (vLLM on LRZ):
  python scripts/generate_paraphrase_only.py \
    --base_url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --input simplification_method/fineweb_source/fineweb_2m_new_source.txt \
    --run_all
"""

import json
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from openai import AsyncOpenAI


PARAPHRASE_SYSTEM = """\
You are paraphrasing web text for a language model training dataset.
For each input passage, output exactly one line: a paraphrase.

Paraphrase rules:
- Preserve the full meaning. Change structure and vocabulary significantly.
- Match the original register and tense.
- Match the original length as closely as possible — do not compress or expand.
- Do NOT add explanations, labels, or meta-commentary.
- Output exactly one line of text, nothing else.\
"""


def is_good_line(line: str) -> bool:
    words = line.split()
    if len(words) < 15:
        return False
    non_ascii = sum(1 for c in line if ord(c) > 127)
    if non_ascii / max(len(line), 1) > 0.1:
        return False
    return True


def line_id(line: str) -> str:
    return hashlib.md5(line.encode()).hexdigest()[:16]


def filter_output(text: str, original: str) -> str | None:
    if not text:
        return None
    text = text.strip().strip('"\'')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    text = lines[0]
    words = text.split()
    orig_words = len(original.split())
    if len(words) < max(8, orig_words * 0.25) or len(words) > orig_words * 3.0:
        return None
    bad = ["paraphrase", "original sentence", "rewritten", "here is", "here's",
           "i have", "the sentence"]
    if any(p in text.lower()[:60] for p in bad):
        return None
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.15:
        return None
    return text


async def call_api(client, model, system, user_content, semaphore,
                    temperature=0.80, max_tokens=800):
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  API error after 3 attempts: {e}")
                    return None


async def run(client, model, args, semaphore):
    input_file = Path(args.input)
    out_jsonl = Path(args.out_jsonl)
    failed_file = Path(args.failed_file)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    good_lines = [l for l in all_lines if is_good_line(l)]
    print(f"[Info] Total lines: {len(all_lines):,} | Good lines: {len(good_lines):,}")

    done: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    if failed_file.exists():
        with open(failed_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    print(f"[Info] Already done: {len(done):,} | Remaining: {len(good_lines) - len(done):,}")

    todo = [(line_id(l), l) for l in good_lines if line_id(l) not in done]
    if not todo:
        print("[Info] Nothing to do.")
        return

    print(f"[Info] Launching {len(todo):,} API requests (paraphrase only)...")

    async def run_one(lid, original):
        # Scale max_tokens to the input length so long documents (up to ~2000
        # words) are not truncated mid-paraphrase. ~1.5 tokens/word + buffer.
        orig_words = len(original.split())
        dynamic_max_tokens = min(4096, max(800, int(orig_words * 1.8) + 200))
        text = await call_api(
            client, model, PARAPHRASE_SYSTEM, original, semaphore,
            temperature=0.80, max_tokens=dynamic_max_tokens,
        )
        return lid, original, text

    coros = [run_one(*t) for t in todo]
    out_f = open(out_jsonl, "a", encoding="utf-8")
    fail_f = open(failed_file, "a", encoding="utf-8")
    passed = 0
    failed_count = 0

    for i, coro in enumerate(asyncio.as_completed(coros)):
        lid, original, text = await coro
        para = filter_output(text, original) if text else None
        if para:
            out_f.write(json.dumps({
                "id": lid, "original": original, "paraphrase": para,
            }, ensure_ascii=False) + "\n")
            passed += 1
        else:
            fail_f.write(json.dumps({
                "id": lid, "original": original, "raw": text or "",
            }, ensure_ascii=False) + "\n")
            failed_count += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{len(coros):,} | passed: {passed:,} | failed: {failed_count:,}")
            out_f.flush()
            fail_f.flush()

    out_f.close()
    fail_f.close()
    print(f"[Done] Passed: {passed:,} | Failed: {failed_count:,}")


def build_text(out_jsonl: str, out_txt: str, include_original: bool = True):
    out_jsonl = Path(out_jsonl)
    out_txt = Path(out_txt)

    import random
    records = []
    with open(out_jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    rng = random.Random(42)
    rng.shuffle(records)

    total_words = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for obj in records:
            orig = obj.get("original", "")
            para = obj.get("paraphrase", "")
            block = []
            if include_original and orig:
                block.append(orig)
            if para:
                block.append(para)
            if block:
                text = " ".join(block)
                f.write(text + "\n\n")
                total_words += sum(len(s.split()) for s in block)

    print(f"[Build] Wrote {len(records):,} groups ({total_words:,} words) to {out_txt}")


async def main_async(args):
    client = AsyncOpenAI(
        api_key=args.api_key or "token-none",
        base_url=args.base_url,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.run_all:
        await run(client, args.model, args, semaphore)
    if args.build_text:
        build_text(args.out_jsonl, args.out_txt, include_original=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="simplification_method/fineweb_source/fineweb_2m_new_source.txt")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--build_text", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out_jsonl", default="data/fineweb_2m_new_paraphrase.jsonl")
    parser.add_argument("--failed_file", default="data/fineweb_2m_new_paraphrase_failed.jsonl")
    parser.add_argument("--out_txt", default="data/fineweb_2m_new_paraphrase.txt")
    args = parser.parse_args()

    if not args.run_all and not args.build_text:
        parser.error("Specify --run_all and/or --build_text")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
