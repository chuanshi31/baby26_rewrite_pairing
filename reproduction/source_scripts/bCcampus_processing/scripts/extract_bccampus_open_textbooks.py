#!/usr/bin/env python3
"""Download and extract natural paragraphs from selected BCcampus textbooks."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


SEARCH_URL = "https://search2.repo.bccws.ca/indexes/resource/search"
DETAIL_URL = "https://api2.repo.bccws.ca/api/resources/{id}/details"
MEILI_KEY = "751961c49d8babb52dc3b185f28c1cd105a20d39e346fd357073b49786b16b5c"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SUBJECTS = (
    "Social Sciences",
    "Humanities",
    "Communication & Writing",
    "Business",
    "Academic/Career Success",
    "Education",
)

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SKIP_PATTERNS = (
    "creative commons",
    "isbn",
    "copyright",
    "license",
    "download for free",
    "versioning history",
    "pressbooks",
    "bccampus",
    "attribution",
    "redistribute",
    "review questions",
    "discussion questions",
    "critical thinking",
    "learning objectives",
    "key takeaways",
    "chapter summary",
    "references",
    "bibliography",
    "retrieved from",
    "figure ",
    "table ",
    "appendix ",
    "long description",
    "activity:",
    "exercise ",
    "exercises",
    "check your understanding",
    "for instructors",
    "student resources",
    "instructor resources",
    "http://",
    "https://",
    "www.",
)


def request_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    headers = dict(HEADERS)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = f"Bearer {MEILI_KEY}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return text[:90] or "resource"


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def should_keep(text: str, min_words: int, max_words: int) -> bool:
    text = normalize(text)
    if not text:
        return False
    lowered = text.lower()
    if any(pattern in lowered for pattern in SKIP_PATTERNS):
        return False
    if text.count("?") >= 2:
        return False
    words = WORD_RE.findall(text)
    if len(words) < min_words or len(words) > max_words:
        return False
    chars = max(len(text), 1)
    if sum(ch.isdigit() for ch in text) / chars > 0.12:
        return False
    if sum(ord(ch) > 127 for ch in text) / chars > 0.12:
        return False
    symbols = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:'\"!?()-/") for ch in text)
    if symbols / chars > 0.05:
        return False
    if not any(mark in text for mark in ".?!"):
        return False
    return True


class PressbooksParser(HTMLParser):
    BLOCK_TAGS = {"p", "li"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, str, str]] = []
        self.current: list[str] = []
        self.blocks: list[str] = []
        self.in_content = 0
        self.in_bad = False
        self.title = ""
        self.license_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: v or "" for k, v in attrs}
        cls = attrs_dict.get("class", "")
        id_ = attrs_dict.get("id", "")
        self.stack.append((tag, cls, id_))
        if tag in {"script", "style", "nav"} or "toc" in id_ or "front-matter" in cls:
            self.in_bad = True
        if tag == "div" and (
            "chapter" in cls
            or "part" in cls
            or "ugc" in cls
            or "entry-content" in cls
        ) and "chapter-title" not in cls:
            self.in_content += 1
        if tag == "meta" and attrs_dict.get("name") == "pb-title":
            self.title = attrs_dict.get("content", "")
        if tag == "a" and "creativecommons.org/licenses" in attrs_dict.get("href", ""):
            self.license_links.append(attrs_dict["href"])

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            text = normalize("".join(self.current))
            if text and self.in_content and not self.in_bad:
                self.blocks.append(text)
            self.current = []
        if self.stack:
            old_tag, cls, _ = self.stack.pop()
            if old_tag == "div" and (
                "chapter" in cls
                or "part" in cls
                or "ugc" in cls
                or "entry-content" in cls
            ) and "chapter-title" not in cls:
                self.in_content = max(0, self.in_content - 1)
        self.in_bad = any(
            tag_ in {"script", "style", "nav"} or "toc" in id_ or "front-matter" in cls
            for tag_, cls, id_ in self.stack
        )

    def handle_data(self, data: str) -> None:
        if not self.in_bad:
            self.current.append(data)


def query_subject(subject: str, hits_per_page: int) -> list[dict[str, object]]:
    payload = {
        "q": "",
        "filter": [
            "type.code = textbook",
            "currentState = published",
            "sites = bcoc",
            f"concepts.subject = '{subject}'",
        ],
        "hitsPerPage": hits_per_page,
    }
    data = request_json(SEARCH_URL, payload)
    hits = data.get("hits", [])
    return hits if isinstance(hits, list) else []


def query_text(query: str, hits_per_page: int) -> list[dict[str, object]]:
    payload = {
        "q": query,
        "filter": [
            "type.code = textbook",
            "currentState = published",
            "sites = bcoc",
        ],
        "hitsPerPage": hits_per_page,
    }
    data = request_json(SEARCH_URL, payload)
    hits = data.get("hits", [])
    return hits if isinstance(hits, list) else []


def instance_url(instance: dict[str, object]) -> str | None:
    url = instance.get("url")
    if isinstance(url, str) and url:
        return url
    file_obj = instance.get("file")
    if isinstance(file_obj, dict):
        file_url = file_obj.get("url")
        if isinstance(file_url, str) and file_url:
            return file_url
    return None


def get_concepts(detail: dict[str, object], scheme: str) -> list[str]:
    concepts = detail.get("concepts")
    if not isinstance(concepts, dict):
        return []
    path = concepts.get(scheme)
    if isinstance(path, dict):
        values = path.get("concepts")
        if isinstance(values, list):
            return [str(v.get("name")) for v in values if isinstance(v, dict) and v.get("name")]
    return []


def find_xhtml_url(detail: dict[str, object]) -> tuple[str | None, str | None]:
    instances = detail.get("instances")
    if not isinstance(instances, list):
        return None, None

    read_online = None
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        label = str(inst.get("label") or "")
        url = instance_url(inst)
        if not url:
            continue
        if "xhtml" in label.lower() or "type=xhtml" in url:
            return url, "explicit-xhtml"
        if "read online" in label.lower():
            read_online = url

    if read_online and re.search(r"(pressbooks\.pub|opentextbc\.ca)", read_online):
        return read_online.rstrip("/") + "/open/download?type=xhtml", "inferred-xhtml"
    return None, read_online


def extract_file(path: Path, min_words: int, max_words: int) -> tuple[list[str], dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parser = PressbooksParser()
    parser.feed(text)
    paragraphs = []
    for block in parser.blocks:
        if should_keep(block, min_words, max_words):
            paragraphs.append(normalize(block))
    return paragraphs, {
        "parser_title": parser.title,
        "license_links": sorted(set(parser.license_links)),
        "raw_blocks": len(parser.blocks),
        "kept_paragraphs": len(paragraphs),
        "kept_words": sum(count_words(p) for p in paragraphs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    parser.add_argument("--hits_per_subject", type=int, default=1000)
    parser.add_argument("--raw_dir", default="data/bccampus_open_textbooks_raw")
    parser.add_argument("--output", default="data/bccampus_open_textbooks_natural.txt")
    parser.add_argument("--manifest", default="data/bccampus_open_textbooks_manifest.json")
    parser.add_argument("--min_words", type=int, default=15)
    parser.add_argument("--max_words", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--queries", nargs="*", default=[])
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--target_words", type=int, default=0)
    parser.add_argument("--exclude_license_contains", nargs="*", default=[])
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    seen_ids: set[int] = set()
    if args.exclude_manifest:
        for exclude_manifest_name in args.exclude_manifest:
            exclude_path = Path(exclude_manifest_name)
            if exclude_path.exists():
                exclude_manifest = json.loads(exclude_path.read_text(encoding="utf-8"))
                for old in exclude_manifest.get("files", []):
                    if isinstance(old, dict) and isinstance(old.get("id"), int):
                        seen_ids.add(old["id"])

    resources: list[dict[str, object]] = []
    subject_counts: dict[str, int] = {}
    for subject in args.subjects:
        hits = query_subject(subject, args.hits_per_subject)
        subject_counts[subject] = len(hits)
        for hit in hits:
            rid = hit.get("id")
            if isinstance(rid, int) and rid not in seen_ids:
                seen_ids.add(rid)
                resources.append({"id": rid, "search_name": hit.get("name"), "search_subject": subject})
    for query in args.queries:
        hits = query_text(query, args.hits_per_subject)
        subject_counts[f"query:{query}"] = len(hits)
        for hit in hits:
            rid = hit.get("id")
            if isinstance(rid, int) and rid not in seen_ids:
                seen_ids.add(rid)
                resources.append({"id": rid, "search_name": hit.get("name"), "search_query": query})

    files_manifest = []
    total_words = 0
    total_paragraphs = 0
    with output.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(resources, 1):
            rid = item["id"]
            print(f"[{idx}/{len(resources)}] resource {rid}: {item.get('search_name')}", flush=True)
            record = dict(item)
            try:
                detail_data = request_json(DETAIL_URL.format(id=rid))
                detail = detail_data.get("data", {})
                if not isinstance(detail, dict):
                    raise ValueError("detail data is not an object")
                record["name"] = detail.get("name")
                record["slug"] = (detail.get("meta") or {}).get("slug") if isinstance(detail.get("meta"), dict) else None
                record["subjects"] = get_concepts(detail, "subject")
                record["licenses"] = get_concepts(detail, "license")
                if args.exclude_license_contains and any(
                    needle.lower() in license_name.lower()
                    for license_name in record["licenses"]
                    for needle in args.exclude_license_contains
                ):
                    record["status"] = "skipped-license"
                    files_manifest.append(record)
                    continue
                xhtml_url, fallback_url = find_xhtml_url(detail)
                record["xhtml_url"] = xhtml_url
                record["fallback_url"] = fallback_url
                if not xhtml_url:
                    record["status"] = "skipped-no-xhtml"
                    files_manifest.append(record)
                    continue

                name = slugify(str(detail.get("name") or rid))
                raw_path = raw_dir / f"{rid}_{name}.xhtml"
                if not raw_path.exists() or raw_path.stat().st_size == 0:
                    raw_path.write_bytes(fetch_bytes(xhtml_url))
                    time.sleep(args.sleep)
                record["raw_path"] = str(raw_path)
                record["raw_size"] = raw_path.stat().st_size

                paragraphs, stats = extract_file(raw_path, args.min_words, args.max_words)
                for paragraph in paragraphs:
                    out.write(paragraph + "\n")
                record.update(stats)
                record["status"] = "ok" if paragraphs else "no-kept-paragraphs"
                total_words += int(stats["kept_words"])
                total_paragraphs += int(stats["kept_paragraphs"])
                if args.target_words and total_words >= args.target_words:
                    files_manifest.append(record)
                    print(f"[Info] target_words reached: {total_words:,}", flush=True)
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
            files_manifest.append(record)

    manifest = {
        "description": "BCcampus Open Collection textbooks extracted from selected OpenStax-like subjects.",
        "subjects": list(args.subjects),
        "queries": list(args.queries),
        "subject_hit_counts": subject_counts,
        "unique_resources": len(resources),
        "output": str(output),
        "raw_dir": str(raw_dir),
        "kept_paragraphs": total_paragraphs,
        "kept_words": total_words,
        "filters": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "skip_patterns": list(SKIP_PATTERNS),
        },
        "files": files_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] {total_paragraphs:,} paragraphs | {total_words:,} words -> {output}")
    print(f"[Done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
