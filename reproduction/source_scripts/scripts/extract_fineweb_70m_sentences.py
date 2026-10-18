#!/usr/bin/env python3
"""Extract a ~70M-word sentence-level FineWeb-Edu slice.

Input documents are expected one per line. Output is one sentence per line,
matching the sentence-level FineWeb files used by the simplification pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“”‘’(\[]?[A-Z0-9])")
ABBREVIATIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "No.",
    "Fig.",
    "vs.",
    "e.g.",
    "i.e.",
    "U.S.",
    "U.K.",
    "D.C.",
)
DOT = "<DOT>"

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
    text = re.sub(r"\s+", " ", text.strip())
    return text


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    normalized = normalize(text)
    protected = normalized
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", DOT))
    parts = SENTENCE_END_RE.split(protected)
    sentences: list[str] = []
    for part in parts:
        sent = part.replace(DOT, ".").strip()
        if not sent:
            continue
        sentences.append(sent)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/fineweb_edu_all.txt")
    parser.add_argument(
        "--output",
        default="simplification_method/fineweb_source/fineweb_70m_sentences.txt",
    )
    parser.add_argument(
        "--manifest",
        default="simplification_method/fineweb_source/fineweb_70m_sentences_manifest.json",
    )
    parser.add_argument("--target_words", type=int, default=70_000_000)
    parser.add_argument("--min_sentence_words", type=int, default=5)
    parser.add_argument("--max_sentence_words", type=int, default=80)
    parser.add_argument("--max_doc_words", type=int, default=1800)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    scanned_docs = 0
    kept_docs = 0
    kept_sentences = 0
    kept_words = 0
    reason_counts: Counter[str] = Counter()
    examples: list[str] = []

    with input_path.open(encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
        for line in inp:
            if kept_words >= args.target_words:
                break
            scanned_docs += 1
            doc = normalize(line)
            if not doc:
                reason_counts["empty_doc"] += 1
                continue
            lowered = doc.lower()
            if any(pattern in lowered for pattern in SKIP_PATTERNS):
                reason_counts["boilerplate_doc"] += 1
                continue
            doc_words = count_words(doc)
            if doc_words > args.max_doc_words:
                reason_counts["too_long_doc"] += 1
                continue

            doc_kept = False
            for sentence in split_sentences(doc):
                words = count_words(sentence)
                if words < args.min_sentence_words:
                    reason_counts["too_short_sentence"] += 1
                    continue
                if words > args.max_sentence_words:
                    reason_counts["too_long_sentence"] += 1
                    continue
                out.write(sentence + "\n")
                kept_sentences += 1
                kept_words += words
                doc_kept = True
                if len(examples) < 12:
                    examples.append(sentence)
                if kept_words >= args.target_words:
                    break
            if doc_kept:
                kept_docs += 1

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "target_words": args.target_words,
        "kept_words": kept_words,
        "scanned_docs": scanned_docs,
        "kept_docs": kept_docs,
        "kept_sentences": kept_sentences,
        "min_sentence_words": args.min_sentence_words,
        "max_sentence_words": args.max_sentence_words,
        "max_doc_words": args.max_doc_words,
        "reason_counts": dict(reason_counts),
        "examples": examples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
