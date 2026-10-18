#!/usr/bin/env python3
"""
Simplification augmentation for openstax / fineweb text files.

For each input line (15-200 words):
  - Generate 1 simplified version (shorter, simpler vocabulary, same core meaning)

Output: data/<name>_simplified.jsonl   (resume-friendly)
        data/<name>_simplified_failed.jsonl
        data/<name>_simplified.txt     (build_text only)

Usage (vLLM on LRZ):
  python scripts/generate_simplification.py \
    --base_url http://localhost:8002/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --input data/openstax_natural_humanities.txt \
    --run_all
"""

import json
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SIMPLIFY_SYSTEM = """\
You are simplifying academic sentences for a language model training dataset.
For each input sentence, output exactly one line: a simplified version.

Simplification rules:
- Keep the core meaning intact.
- Use simpler, more common vocabulary. Replace jargon with plain words.
- Shorten where possible, but do not omit key information.
- Write in a clear, direct style accessible to a general audience.
- Do NOT add explanations, labels, or meta-commentary.
- Output exactly one line of text, nothing else.\
"""


# ---------------------------------------------------------------------------
# Line filtering
# ---------------------------------------------------------------------------

def is_good_line(line: str) -> bool:
    words = line.split()
    if len(words) < 15:
        return False
    non_ascii = sum(1 for c in line if ord(c) > 127)
    if non_ascii / max(len(line), 1) > 0.1:
        return False
    if re.match(r'^[a-z ]{2,40}:\s', line) and len(words) < 20:
        return False
    if re.search(r'\d+\s*[•·]\s*\w', line):
        return False
    return True


def line_id(line: str) -> str:
    return hashlib.md5(line.encode()).hexdigest()[:16]


def filter_output(text: str, original: str) -> str | None:
    if not text:
        return None
    text = text.strip().strip('"\'')
    # Must be a single line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) == 0:
        return None
    text = lines[0]  # take first line if multiple
    words = text.split()
    orig_words = len(original.split())
    # At least 5 words; no upper limit (original can be very long)
    if len(words) < 5:
        return None
    bad = ["simplif", "paraphrase", "original sentence", "here is", "here's",
           "i have", "the sentence", "rewritten", "in simpler terms:"]
    if any(p in text.lower() for p in bad):
        return None
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.15:
        return None
    return text


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

async def call_api(client, model, system, user_content, semaphore,
                   temperature=0.70, max_tokens=300):
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


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

async def run(client, model, args, semaphore):
    input_file = Path(args.input)
    out_jsonl = Path(args.out_jsonl)
    failed_file = Path(args.failed_file)

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

    print(f"[Info] Launching {len(todo):,} API requests (simplification)...")

    async def run_one(lid, original):
        text = await call_api(
            client, model, SIMPLIFY_SYSTEM,
            original,
            semaphore,
            temperature=0.70,
            max_tokens=600,
        )
        return lid, original, text

    coros = [run_one(*t) for t in todo]
    out_f = open(out_jsonl, "a", encoding="utf-8")
    fail_f = open(failed_file, "a", encoding="utf-8")
    passed = 0
    failed_count = 0

    for i, coro in enumerate(asyncio.as_completed(coros)):
        lid, original, text = await coro
        simplified = filter_output(text, original) if text else None
        if simplified:
            out_f.write(json.dumps({
                "id": lid,
                "original": original,
                "simplified": simplified,
            }, ensure_ascii=False) + "\n")
            passed += 1
        else:
            fail_f.write(json.dumps({
                "id": lid,
                "original": original,
                "raw": text or "",
            }, ensure_ascii=False) + "\n")
            failed_count += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1:,}/{len(coros):,} | passed: {passed:,} | failed: {failed_count:,}")
            out_f.flush()
            fail_f.flush()

    out_f.close()
    fail_f.close()
    print(f"[Done] Passed: {passed:,} | Failed: {failed_count:,}")


# ---------------------------------------------------------------------------
# Build text file
# ---------------------------------------------------------------------------

def build_text(out_jsonl: str, out_txt: str, include_original: bool = True):
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
            simp = obj.get("simplified", "")
            parts = []
            if include_original and orig:
                parts.append(orig)
            if simp:
                parts.append(simp)
            if parts:
                line = " ".join(parts)
                f.write(line + "\n")
                total_words += len(line.split())

    print(f"[Build] Wrote {len(records):,} records ({total_words:,} words) to {out_txt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--base_url", default="http://localhost:8002/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--build_text", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--failed_file", required=True)
    parser.add_argument("--out_txt", required=True)
    args = parser.parse_args()

    if not args.run_all and not args.build_text:
        parser.error("Specify --run_all and/or --build_text")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
