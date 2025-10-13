from __future__ import annotations
#!/usr/bin/env python3
"""
Extract fields from Runtime/GSM8k/outputs-qwen3-8B/train/*.txt

For each file, extract:
  - index (int)
  - duration_seconds (float)
  - prompt (string)  — the text between '=== prompt' header and 'length of question' (or next header)
    - length_of_question (int)
  - length of prompt_token_ids (int)
  - length of output_token_ids (int)
  - full_response (string) — content after '= full_response =' up to '= length of full_response =' or EOF

Outputs JSONL by default; optional CSV.
"""
"""
python Runtime/scripts/extract_train_outputs.py --input-dir Runtime/GSM8k/outputs-qwen3-8B/train --out-jsonl Runtime/GSM8k/outputs-qwen3-8B/train_extracted.jsonl
"""


import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple


TRAIN_DIR_DEFAULT = Path("Runtime/GSM8k/outputs-qwen3-8B/train")
JSONL_DEFAULT = TRAIN_DIR_DEFAULT.parent / "train_extracted.jsonl"


@dataclass
class Record:
    index: Optional[int]
    duration_seconds: Optional[float]
    prompt: Optional[str]
    length_of_question: Optional[int]
    length_of_prompt_token_ids: Optional[int]
    length_of_output_token_ids: Optional[int]
    full_response: Optional[str]
    # Optional: source file for traceability
    source_file: str


def _find_first_line_idx(lines: List[str], pattern: re.Pattern) -> int:
    for i, ln in enumerate(lines):
        if pattern.search(ln):
            return i
    return -1


def _extract_between(lines: List[str], start_idx: int, end_patterns: List[re.Pattern]) -> Tuple[str, int]:
    """Return text between start_idx (exclusive) and first match of any end pattern (exclusive).
    If not found, consume to EOF. Returns (text, end_idx) where end_idx is the index of the end line or len(lines).
    """
    # content starts after start_idx
    s = start_idx + 1
    e = len(lines)
    for i in range(s, len(lines)):
        ln = lines[i]
        for pat in end_patterns:
            if pat.search(ln):
                e = i
                return "\n".join([l.rstrip("\n") for l in lines[s:e]]).rstrip("\n"), e
    # if no end found
    return "\n".join([l.rstrip("\n") for l in lines[s:]]).rstrip("\n"), len(lines)


def parse_file(path: Path) -> Record:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # index
    index = None
    m = re.search(r"^index:\s*(\d+)", text, flags=re.MULTILINE)
    if m:
        try:
            index = int(m.group(1))
        except Exception:
            index = None

    # duration_seconds
    duration_seconds = None
    m = re.search(r"^duration_seconds:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.MULTILINE)
    if m:
        try:
            duration_seconds = float(m.group(1))
        except Exception:
            duration_seconds = None

    # prompt
    # look for a header line starting with '=== prompt'
    prompt_header_pat = re.compile(r"^===\s*prompt.*===\s*$")
    prompt_len_line_pat = re.compile(r"^length of question\s*=", re.IGNORECASE)
    next_section_pat = re.compile(r"^===\s*model_output\s*===\s*$")
    start_idx = _find_first_line_idx(lines, prompt_header_pat)
    prompt_text: Optional[str] = None
    if start_idx != -1:
        prompt_text, _ = _extract_between(
            lines,
            start_idx,
            [prompt_len_line_pat, next_section_pat],
        )

    # length_of_question
    length_of_question: Optional[int] = None
    m = re.search(r"^length of question\s*=\s*(\d+)", text, flags=re.MULTILINE | re.IGNORECASE)
    if m:
        try:
            length_of_question = int(m.group(1))
        except Exception:
            length_of_question = None

    # length_of_prompt_token_ids
    lpt_len = None
    lpt_header_pat = re.compile(r"^=\s*length of prompt_token_ids\s*=\s*$", re.IGNORECASE)
    i = _find_first_line_idx(lines, lpt_header_pat)
    if i != -1 and i + 1 < len(lines):
        try:
            lpt_len = int(lines[i + 1].strip())
        except Exception:
            lpt_len = None

    # length_of_output_token_ids
    lot_len = None
    lot_header_pat = re.compile(r"^=\s*length of output_token_ids\s*=\s*$", re.IGNORECASE)
    i = _find_first_line_idx(lines, lot_header_pat)
    if i != -1 and i + 1 < len(lines):
        try:
            lot_len = int(lines[i + 1].strip())
        except Exception:
            lot_len = None

    # full_response
    fr_text: Optional[str] = None
    fr_header_pat = re.compile(r"^=\s*full_response\s*=\s*$", re.IGNORECASE)
    fr_end_pat = re.compile(r"^=\s*length of full_response\s*=\s*$", re.IGNORECASE)
    i = _find_first_line_idx(lines, fr_header_pat)
    if i != -1:
        fr_text, _ = _extract_between(lines, i, [fr_end_pat])

    return Record(
        index=index,
        duration_seconds=duration_seconds,
        prompt=prompt_text,
        length_of_question=length_of_question,
        length_of_prompt_token_ids=lpt_len,
        length_of_output_token_ids=lot_len,
        full_response=fr_text,
        source_file=str(path),
    )


def write_jsonl(records: List[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_csv(records: List[Record], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "duration_seconds",
        "length_of_question",
        "length_of_prompt_token_ids",
        "length_of_output_token_ids",
        "prompt",
        "full_response",
        "source_file",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def collect_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.txt") if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="Extract GSM8k outputs to JSONL/CSV")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TRAIN_DIR_DEFAULT,
        help=f"Input directory containing txt files (default: {TRAIN_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=JSONL_DEFAULT,
        help=f"Output JSONL path (default: {JSONL_DEFAULT})",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    files = collect_files(args.input_dir)
    if not files:
        print(f"No .txt files found in {args.input_dir}")
        return 1

    records: List[Record] = []
    for p in files:
        try:
            rec = parse_file(p)
        except Exception as e:
            # Create a minimal record indicating failure
            rec = Record(
                index=None,
                duration_seconds=None,
                prompt=None,
                length_of_prompt_token_ids=None,
                length_of_output_token_ids=None,
                full_response=f"<parse_error> {e}",
                source_file=str(p),
            )
        records.append(rec)

    write_jsonl(records, args.out_jsonl)
    if args.out_csv is not None:
        write_csv(records, args.out_csv)

    print(
        f"Wrote {len(records)} records to {args.out_jsonl}"
        + (f" and {args.out_csv}" if args.out_csv else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
