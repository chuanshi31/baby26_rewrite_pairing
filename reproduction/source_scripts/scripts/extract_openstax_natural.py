#!/usr/bin/env python3
"""Build a natural-language OpenStax subset for BabyLM-style pretraining."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NATURAL_TEXTBOOK_FILES = (
    "AmericanGovernment3e-WEB.txt",
    "BusinessEthics-OP.txt",
    "Business_Law_I_Essentials_-_WEB.txt",
    "College_Success_-_WEB_uM60O3O.txt",
    "College_Success_Concise.txt",
    "Entrepreneurship-WEB_qNSArxT.txt",
    "Introduction_to_Intellectual_Property_2PYaEdp.txt",
    "Introduction_to_Philosophy-WEB_cszrKYp.txt",
    "Introduction_to_Political_Science_-_WEB.txt",
    "IntroductiontoAnthropology-WEB.txt",
    "IntroductiontoSociology3e-WEB_9QTqRGQ.txt",
    "OrganizationalBehavior-OP_TtwWIeQ.txt",
    "Preparing-for-College-Success-WEB.txt",
    "Principles_Marketing-WEB.txt",
    "PrinciplesofManagement-OP_mGBMvoU.txt",
    "Psychology2e_WEB.txt",
    "USHistory-WEB.txt",
    "World_History_Volume_1-WEB.txt",
    "World_History_Volume_2-WEB_LdwoslB.txt",
    "WritingGuide-WEB.txt",
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

SKIP_PATTERNS = (
    "openstax",
    "creative commons",
    "rice university",
    "isbn",
    "copyright",
    "license",
    "senior contributing author",
    "contributing author",
    "reviewer",
    "mission to transform learning",
    "barriers to learning",
    "educational goals",
    "remix the content",
    "assigning your students",
    "direct link",
    "web view",
    "errata",
    "subject matter experts",
    "past errata",
    "about american government",
    "changes to the",
    "engaging features",
    "module materials",
    "student and instructor resources",
    "instructor resources",
    "community hubs",
    "technology partners",
    "access for free",
    "preface",
    "individual print copies",
    "bulk orders",
    "attribution",
    "redistribute",
    "digital format",
    "print format",
    "physical page",
    "bibliographic reference",
    "nonprofit ed tech",
    "supporter community",
    "foundation",
    "access. the future of education",
    "online learning platform",
    "low-cost",
    "chapter outline",
    "learning objectives",
    "key terms",
    "review questions",
    "critical thinking questions",
    "references",
    "bibliography",
    "career connection",
    "chapter review",
    "answer key",
    "chapter ",
    "retrieved from",
    "proceedings of",
    "journal of",
    "university press",
    "bantam books",
    "bbc radio",
    "figure ",
    "table ",
    "visit ",
    "website",
    "http://",
    "https://",
    "www.",
)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def iter_paragraphs(lines: list[str], start: int) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []
    for raw_line in lines[start:]:
        line = normalize_line(raw_line)
        if not line:
            if buf:
                paragraphs.append(normalize_line(" ".join(buf)))
                buf = []
            continue
        # Page numbers and bullet-only lines are layout artifacts.
        if re.fullmatch(r"\d+", line) or line in {"•", "◦"}:
            continue
        buf.append(line)
        if re.search(r'[.!?]["”’)]?$', line):
            paragraphs.append(normalize_line(" ".join(buf)))
            buf = []
    if buf:
        paragraphs.append(normalize_line(" ".join(buf)))
    return paragraphs


def should_keep_paragraph(text: str, min_words: int, max_words: int) -> bool:
    text = normalize_line(text)
    if not text:
        return False
    lowered = text.lower()
    if any(pattern in lowered for pattern in SKIP_PATTERNS):
        return False
    if re.match(r"^\d+(\.\d+)+\s+[A-Z]", text):
        return False
    if re.match(r"^\d+\.\s+[A-Z]", text):
        return False
    if re.search(r"\b\d+\.\s+[A-D]\b", text):
        return False
    if text.count("?") >= 2:
        return False

    words = WORD_RE.findall(text)
    word_count = len(words)
    if word_count < min_words or word_count > max_words:
        return False

    chars = len(text)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(chars, 1)
    if digit_ratio > 0.12:
        return False

    symbols = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:'\"!?()-") for ch in text)
    symbol_ratio = symbols / max(chars, 1)
    if symbol_ratio > 0.05:
        return False

    alpha = sum(ch.isalpha() for ch in text)
    uppercase = sum(ch.isupper() for ch in text)
    upper_ratio = uppercase / max(alpha, 1)
    if upper_ratio > 0.70 and word_count < 12:
        return False

    # Keep paragraphs with sentence-like punctuation.
    if not any(mark in text for mark in ".?!"):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/openstax-text/data")
    parser.add_argument("--output", default="data/openstax_natural_humanities.txt")
    parser.add_argument("--manifest", default="data/openstax_natural_humanities_manifest.json")
    parser.add_argument("--min_words", type=int, default=8)
    parser.add_argument("--max_words", type=int, default=180)
    parser.add_argument("--skip_initial_lines", type=int, default=900)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files_manifest: list[dict[str, object]] = []
    total_source_words = 0
    kept_words = 0
    kept_lines = 0

    with output.open("w", encoding="utf-8") as out:
        for filename in NATURAL_TEXTBOOK_FILES:
            path = input_dir / filename
            if not path.exists():
                files_manifest.append({"file": filename, "missing": True})
                continue

            text = path.read_text(encoding="utf-8", errors="ignore")
            source_words = count_words(text)
            total_source_words += source_words

            file_kept_words = 0
            file_kept_lines = 0
            paragraphs = iter_paragraphs(text.splitlines(), args.skip_initial_lines)
            for paragraph in paragraphs:
                if not should_keep_paragraph(paragraph, args.min_words, args.max_words):
                    continue
                out.write(paragraph + "\n")
                paragraph_words = count_words(paragraph)
                file_kept_words += paragraph_words
                file_kept_lines += 1

            kept_words += file_kept_words
            kept_lines += file_kept_lines
            files_manifest.append(
                {
                    "file": filename,
                    "source_words": source_words,
                    "kept_words": file_kept_words,
                    "kept_paragraphs": file_kept_lines,
                }
            )

    manifest = {
        "description": "Natural-language OpenStax subset: humanities, social science, writing, college success, and management-oriented textbooks.",
        "input_dir": str(input_dir),
        "output": str(output),
        "selected_files": list(NATURAL_TEXTBOOK_FILES),
        "source_words": total_source_words,
        "kept_words": kept_words,
        "kept_paragraphs": kept_lines,
        "filters": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "skip_patterns": list(SKIP_PATTERNS),
            "max_digit_ratio": 0.12,
            "max_symbol_ratio": 0.05,
            "drop_short_all_caps": True,
            "paragraph_reconstruction": True,
            "skip_initial_lines_per_file": args.skip_initial_lines,
        },
        "files": files_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_files": len(NATURAL_TEXTBOOK_FILES),
                "source_words": total_source_words,
                "kept_words": kept_words,
                "kept_lines": kept_lines,
                "output": str(output),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
