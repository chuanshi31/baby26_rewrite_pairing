#!/usr/bin/env python3
"""Extract a fresh ~3M-word FineWeb-Edu slice, excluding docs already used in
fineweb_5m_old_source.txt and fineweb_2m_new_source.txt so the new batch can be
combined with both into a 10M corpus without overlap."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

SKIP_PATTERNS = (
    "cookie policy",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "subscribe",
    "advertisement",
    "comments",
    "posted by",
    "click here",
    "download pdf",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def load_used_prefixes(paths: list[Path], prefix_words: int = 12) -> set[str]:
    """Build a set of normalized first-N-word prefixes for already-used docs."""
    used = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            words = WORD_RE.findall(line.lower())
            if words:
                used.add(" ".join(words[:prefix_words]))
    return used


def keep_text(text: str, score: float, min_score: float, min_words: int, max_words: int, token_count: int) -> tuple[bool, str]:
    lowered = text.lower()
    if score < min_score:
        return False, "low_score"
    words = count_words(text)
    if words < min_words:
        return False, "too_short"
    if words > max_words or token_count > max_words * 2:
        return False, "too_long"
    if any(pattern in lowered for pattern in SKIP_PATTERNS):
        return False, "boilerplate"
    return True, "kept"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/fineweb-edu-sample/sample/10BT/000_00000.parquet")
    parser.add_argument("--output", default="simplification_method/fineweb_source/fineweb_3m_new_source.txt")
    parser.add_argument("--manifest", default="simplification_method/fineweb_source/fineweb_3m_new_source_manifest.json")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[
            "simplification_method/fineweb_source/fineweb_5m_old_source.txt",
            "simplification_method/fineweb_source/fineweb_2m_new_source.txt",
        ],
    )
    parser.add_argument("--min_score", type=float, default=2.5)
    parser.add_argument("--min_words", type=int, default=40)
    parser.add_argument("--max_words", type=int, default=1800)
    parser.add_argument("--target_words", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_paths = [Path(p) for p in args.exclude]
    used_prefixes = load_used_prefixes(exclude_paths)
    print(f"Loaded {len(used_prefixes)} excluded doc prefixes from {exclude_paths}")

    parquet_file = pq.ParquetFile(input_path)
    reason_counts: Counter[str] = Counter()
    kept_docs = 0
    kept_words = 0
    scanned_docs = 0
    examples: list[dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as out:
        for group_idx in range(parquet_file.num_row_groups):
            if kept_words >= args.target_words:
                break
            table = parquet_file.read_row_group(
                group_idx,
                columns=["text", "url", "language", "language_score", "token_count", "score", "int_score"],
            )
            for row in table.to_pylist():
                if kept_words >= args.target_words:
                    break
                scanned_docs += 1
                if row.get("language") != "en":
                    reason_counts["non_english"] += 1
                    continue

                text = normalize(row.get("text") or "")
                token_count = int(row.get("token_count") or 0)
                score = float(row.get("score") or 0.0)
                keep, reason = keep_text(text, score, args.min_score, args.min_words, args.max_words, token_count)
                reason_counts[reason] += 1
                if not keep:
                    continue

                words_list = WORD_RE.findall(text.lower())
                prefix = " ".join(words_list[:12])
                if prefix in used_prefixes:
                    reason_counts["already_used"] += 1
                    continue

                out.write(text + "\n")
                used_prefixes.add(prefix)
                kept_docs += 1
                words = count_words(text)
                kept_words += words
                if len(examples) < 10:
                    examples.append(
                        {
                            "score": score,
                            "int_score": row.get("int_score"),
                            "url": row.get("url"),
                            "text_preview": text[:300],
                        }
                    )

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "excluded_files": [str(p) for p in exclude_paths],
        "scanned_docs": scanned_docs,
        "kept_docs": kept_docs,
        "kept_words": kept_words,
        "target_words": args.target_words,
        "min_score": args.min_score,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "seed": args.seed,
        "reason_counts": dict(reason_counts),
        "examples": examples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scanned_docs": scanned_docs, "kept_docs": kept_docs, "kept_words": kept_words}, indent=2))


if __name__ == "__main__":
    main()
