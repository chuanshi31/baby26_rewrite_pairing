#!/usr/bin/env python3
"""Extract a fresh ~2M-word Simple Wikipedia sentence batch from the larger
paragraph pool, excluding sentences already used in simple_wiki_sentences.txt
so the new batch can be combined with the existing 7M without overlap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nltk

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs", default="simplification_method/simple_wiki_source/simple_wikipedia_paragraphs.txt")
    parser.add_argument(
        "--existing_sentences",
        nargs="+",
        default=["simplification_method/simple_wiki_source/simple_wiki_sentences.txt"],
        help="One or more files of already-used sentences to exclude.",
    )
    parser.add_argument("--output", default="simplification_method/simple_wiki_source/simple_wiki_sentences_2m_new.txt")
    parser.add_argument("--manifest", default="simplification_method/simple_wiki_source/simple_wiki_sentences_2m_new_manifest.json")
    parser.add_argument(
        "--used_sentences_log",
        default="simplification_method/simple_wiki_source/simple_wiki_sentences_used_log.txt",
        help="Cumulative log of every sentence ever extracted by this script, appended to after each run.",
    )
    parser.add_argument("--target_words", type=int, default=2_000_000)
    parser.add_argument("--min_words", type=int, default=4)
    parser.add_argument("--max_words", type=int, default=120)
    args = parser.parse_args()

    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    for src in args.existing_sentences:
        src_path = Path(src)
        if not src_path.exists():
            continue
        for line in src_path.read_text(encoding="utf-8").splitlines():
            line = normalize(line)
            if line:
                existing.add(line)
    used_log_path = Path(args.used_sentences_log)
    if used_log_path.exists():
        for line in used_log_path.read_text(encoding="utf-8").splitlines():
            line = normalize(line)
            if line:
                existing.add(line)
    print(f"Loaded {len(existing)} existing sentences to exclude (incl. cumulative used-log)")

    kept_words = 0
    kept_sents = 0
    scanned_sents = 0
    dup_count = 0
    examples: list[str] = []

    with output_path.open("w", encoding="utf-8") as out, used_log_path.open("a", encoding="utf-8") as used_log:
        for para in Path(args.paragraphs).read_text(encoding="utf-8").splitlines():
            if kept_words >= args.target_words:
                break
            para = para.strip()
            if not para:
                continue
            for sent in nltk.sent_tokenize(para):
                if kept_words >= args.target_words:
                    break
                sent = normalize(sent)
                scanned_sents += 1
                if not sent:
                    continue
                if sent in existing:
                    dup_count += 1
                    continue
                words = count_words(sent)
                if words < args.min_words or words > args.max_words:
                    continue

                out.write(sent + "\n")
                used_log.write(sent + "\n")
                existing.add(sent)
                kept_sents += 1
                kept_words += words
                if len(examples) < 10:
                    examples.append(sent)

    manifest = {
        "paragraphs_source": args.paragraphs,
        "existing_sentences": args.existing_sentences,
        "used_sentences_log": str(used_log_path),
        "output": str(output_path),
        "scanned_sents": scanned_sents,
        "kept_sents": kept_sents,
        "kept_words": kept_words,
        "duplicate_sents_skipped": dup_count,
        "target_words": args.target_words,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "examples": examples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"scanned_sents": scanned_sents, "kept_sents": kept_sents, "kept_words": kept_words, "dup_skipped": dup_count}, indent=2))


if __name__ == "__main__":
    main()
