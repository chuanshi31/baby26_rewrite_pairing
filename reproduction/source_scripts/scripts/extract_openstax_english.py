#!/usr/bin/env python3
"""Extract English OpenStax textbook text into one corpus file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NON_EN_NAME_MARKERS = (
    "calculo",
    "cálculo",
    "fisica",
    "física",
    "fizyka",
    "quimica",
    "química",
    "introduccion",
    "introducción",
    "estadistica",
    "statystyka",
    "universitaria",
    "volumen",
    "tom_",
    "dla_szko",
    "szkół",
    "wstep",
    "wstęp",
)

COMMON_EN = set(
    "the of and to in a is that for with as on are by from or an be this which "
    "at it not have can their more has was were will would about into people "
    "one all when what how why these those"
    .split()
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def looks_english_text(
    text: str,
    min_ascii_alpha_ratio: float,
    min_common_word_ratio: float,
) -> tuple[bool, dict[str, float]]:
    sample = text[:200_000]
    words = [word.lower() for word in WORD_RE.findall(sample)]
    letters = sum(ch.isalpha() for ch in sample)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in sample)
    ascii_alpha_ratio = ascii_letters / max(letters, 1)
    common_word_ratio = sum(word in COMMON_EN for word in words) / max(len(words), 1)
    ok = (
        len(words) > 200
        and ascii_alpha_ratio >= min_ascii_alpha_ratio
        and common_word_ratio >= min_common_word_ratio
    )
    return ok, {
        "ascii_alpha_ratio": ascii_alpha_ratio,
        "common_word_ratio": common_word_ratio,
        "sample_words": float(len(words)),
    }


def should_keep_line(line: str, min_line_words: int) -> bool:
    line = normalize_line(line)
    if not line:
        return False
    words = WORD_RE.findall(line)
    if len(words) < min_line_words:
        return False
    if line.count("|") or line.count("{") or line.count("}") or line.count("=") > 1:
        return False
    digit_ratio = sum(ch.isdigit() for ch in line) / max(len(line), 1)
    if digit_ratio > 0.18:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/openstax-text/data")
    parser.add_argument("--output", default="data/openstax_english.txt")
    parser.add_argument("--manifest", default="data/openstax_english_manifest.json")
    parser.add_argument("--min_ascii_alpha_ratio", type=float, default=0.96)
    parser.add_argument("--min_common_word_ratio", type=float, default=0.055)
    parser.add_argument("--min_line_words", type=int, default=3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    kept_files: list[dict[str, object]] = []
    rejected_files: list[dict[str, object]] = []
    total_words = 0
    kept_words_before_line_filter = 0
    kept_words_after_line_filter = 0
    kept_lines = 0

    with output.open("w", encoding="utf-8") as out:
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            words = count_words(text)
            total_words += words
            lower_name = path.name.lower()
            if any(marker in lower_name for marker in NON_EN_NAME_MARKERS):
                rejected_files.append(
                    {"file": path.name, "words": words, "reason": "filename_non_english"}
                )
                continue

            is_english, scores = looks_english_text(
                text,
                args.min_ascii_alpha_ratio,
                args.min_common_word_ratio,
            )
            if not is_english:
                rejected_files.append(
                    {
                        "file": path.name,
                        "words": words,
                        "reason": "language_heuristic",
                        **scores,
                    }
                )
                continue

            file_words_after_line_filter = 0
            file_lines = 0
            for raw_line in text.splitlines():
                line = normalize_line(raw_line)
                if not should_keep_line(line, args.min_line_words):
                    continue
                out.write(line + "\n")
                line_words = count_words(line)
                file_words_after_line_filter += line_words
                file_lines += 1

            kept_words_before_line_filter += words
            kept_words_after_line_filter += file_words_after_line_filter
            kept_lines += file_lines
            kept_files.append(
                {
                    "file": path.name,
                    "words_before_line_filter": words,
                    "words_after_line_filter": file_words_after_line_filter,
                    "lines_after_line_filter": file_lines,
                    **scores,
                }
            )

    manifest = {
        "input_dir": str(input_dir),
        "output": str(output),
        "files_total": len(files),
        "files_kept": len(kept_files),
        "files_rejected": len(rejected_files),
        "total_words_before_filter": total_words,
        "kept_words_before_line_filter": kept_words_before_line_filter,
        "kept_words_after_line_filter": kept_words_after_line_filter,
        "kept_lines_after_line_filter": kept_lines,
        "filters": {
            "non_en_name_markers": list(NON_EN_NAME_MARKERS),
            "min_ascii_alpha_ratio": args.min_ascii_alpha_ratio,
            "min_common_word_ratio": args.min_common_word_ratio,
            "min_line_words": args.min_line_words,
        },
        "kept_files": kept_files,
        "rejected_files": rejected_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in (
        "files_total",
        "files_kept",
        "files_rejected",
        "total_words_before_filter",
        "kept_words_before_line_filter",
        "kept_words_after_line_filter",
        "kept_lines_after_line_filter",
    )}, indent=2))


if __name__ == "__main__":
    main()
