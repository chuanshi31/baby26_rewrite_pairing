#!/usr/bin/env python3
"""
Paraphrase + Negation augmentation for openstax_natural_humanities.txt

For each input line (15-200 words):
  - Generate 1 paraphrase
  - Generate 1 negation

Output: data/openstax_paraphrase_negation.txt
        data/openstax_paraphrase_negation_paragraphs.jsonl  (for resume)
        data/openstax_paraphrase_negation_failed.jsonl

Usage (vLLM on LRZ):
  python scripts/generate_paraphrase_negation.py \
    --base_url http://localhost:8000/v1 \
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
# Prompts
# ---------------------------------------------------------------------------

COMBINED_SYSTEM = """\
You are augmenting academic sentences for a language model training dataset.
For each input sentence, output exactly two lines:
  Line 1 — PARAPHRASE: rewrite the sentence with different structure and vocabulary, same meaning.
  Line 2 — NEGATION: a naturally flowing sentence that negates or contrasts the original claim.

Output format (no labels, no extra text, exactly 2 lines):
<paraphrase>
<negation>

Paraphrase rules:
- Preserve the full meaning. Change structure and vocabulary significantly.
- Keep academic register. Match the original length as closely as possible.
- Tense must match the original.

Negation rules:
- Do NOT just add "not". Create a meaningful contrastive statement.
- NEVER start with "It is not the case that". Vary your sentence openings every time.
- Use a wide range of patterns, for example:
    "Although X is often assumed, Y is equally important..."
    "Contrary to popular belief, X does not necessarily..."
    "While X plays a role, overlooking Y leads to an incomplete picture."
    "X alone fails to account for...; other factors such as Y are also at work."
    "Rather than X, what actually drives this is Y."
    "The assumption that X ignores the equally significant role of Y."
    "Even if X, this does not mean that Y follows automatically."
    "X may be true in some cases, but Y presents a compelling counterexample."
- Each negation should feel like a distinct, independently written sentence.
- Match the original length as closely as possible — do not compress into a short sentence.
- Must be a complete, grammatical sentence that stands alone.\
"""


# ---------------------------------------------------------------------------
# Line filtering
# ---------------------------------------------------------------------------

def is_good_line(line: str) -> bool:
    words = line.split()
    if len(words) < 15:
        return False
    # Skip lines that are mostly non-ASCII (equations, foreign text)
    non_ascii = sum(1 for c in line if ord(c) > 127)
    if non_ascii / max(len(line), 1) > 0.1:
        return False
    # Skip glossary-style single definitions (word + colon + short def)
    if re.match(r'^[a-z ]{2,40}:\s', line) and len(words) < 20:
        return False
    # Skip lines that look like headers or page references
    if re.search(r'\d+\s*[•·]\s*\w', line):
        return False
    return True


def line_id(line: str) -> str:
    return hashlib.md5(line.encode()).hexdigest()[:16]


def filter_output(text: str, original: str) -> str | None:
    if not text:
        return None
    text = text.strip().strip('"\'')
    words = text.split()
    orig_words = len(original.split())
    # Length check: must be between 50% and 200% of original
    if len(words) < max(8, orig_words * 0.25) or len(words) > orig_words * 3.0:
        return None
    # Must not contain meta-commentary
    bad = ["paraphrase", "negation", "original sentence", "rewritten",
           "here is", "here's", "i have", "the sentence"]
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
                   temperature=0.80, max_tokens=300):
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

    # Load input lines
    with open(input_file, encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    good_lines = [l for l in all_lines if is_good_line(l)]
    print(f"[Info] Total lines: {len(all_lines):,} | Good lines: {len(good_lines):,}")

    # Load already done (keyed by id only — one call per line)
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

    print(f"[Info] Launching {len(todo):,} API requests (paraphrase+negation combined)...")

    async def run_one(lid, original):
        text = await call_api(
            client, model, COMBINED_SYSTEM,
            original,
            semaphore,
            temperature=0.82,
            max_tokens=800,
        )
        return lid, original, text

    coros = [run_one(*t) for t in todo]
    out_f = open(out_jsonl, "a", encoding="utf-8")
    fail_f = open(failed_file, "a", encoding="utf-8")
    passed = 0
    failed_count = 0

    for i, coro in enumerate(asyncio.as_completed(coros)):
        lid, original, text = await coro
        para = neg = ""
        ok = False
        if text:
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            if len(lines) >= 2:
                para = filter_output(lines[0], original)
                neg = filter_output(lines[1], original)
            if para and neg:
                out_f.write(json.dumps({
                    "id": lid,
                    "original": original,
                    "paraphrase": para,
                    "negation": neg,
                }, ensure_ascii=False) + "\n")
                passed += 1
                ok = True
        if not ok:
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
            neg = obj.get("negation", "")

            block = []
            if include_original and orig:
                block.append(orig)
            if para:
                block.append(para)
            if neg:
                block.append(neg)

            if block:
                text = " ".join(block)
                f.write(text + "\n\n")
                total_words += sum(len(s.split()) for s in block)

    print(f"[Build] Wrote {len(all_ids):,} groups ({total_words:,} words) to {out_txt}")


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
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="data/openstax_natural_humanities.txt")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--build_text", action="store_true")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out_jsonl",
                        default="data/openstax_paraphrase_negation.jsonl")
    parser.add_argument("--failed_file",
                        default="data/openstax_paraphrase_negation_failed.jsonl")
    parser.add_argument("--out_txt",
                        default="data/openstax_paraphrase_negation.txt")
    args = parser.parse_args()

    if not args.run_all and not args.build_text:
        parser.error("Specify --run_all and/or --build_text")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
