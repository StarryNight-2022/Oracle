from __future__ import annotations
#!/usr/bin/env python3
"""
Merge all JSONL files in a given directory into a single JSONL file.

Usage examples:
python Runtime/scripts/extract_output_jsonl.py --input-dir Runtime/outputs/outputs-qwen3-0.6B/train_temp_0_no_think --out-jsonl Runtime/outputs/outputs-qwen3-0.6B/train_temp_0_no_think_merged.jsonl

Options:
  --recursive     Recurse into subdirectories
  --sort          Sort by 'name' (default) or 'mtime'
  --skip-empty    Skip empty/whitespace-only lines
  --encoding      File encoding (default: utf-8)
  --ignore-errors Continue on read errors (log and skip)
"""

import argparse
from pathlib import Path
from typing import Iterable, List


def collect_jsonl_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in input_dir.glob(pattern) if p.is_file()]
    return files


def sort_files(files: List[Path], method: str = "name") -> List[Path]:
    if method == "mtime":
        return sorted(files, key=lambda p: (p.stat().st_mtime, str(p)))
    # default: name
    return sorted(files, key=lambda p: str(p))


def merge_jsonl(
    files: Iterable[Path],
    out_path: Path,
    *,
    skip_empty: bool = True,
    encoding: str = "utf-8",
    ignore_errors: bool = True,
) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    written_lines = 0
    with out_path.open("w", encoding=encoding) as fout:
        for fp in files:
            try:
                with fp.open("r", encoding=encoding, errors="replace") as fin:
                    for line in fin:
                        total_lines += 1
                        if skip_empty and not line.strip():
                            continue
                        # Ensure newline termination
                        if not line.endswith("\n"):
                            line = line + "\n"
                        fout.write(line)
                        written_lines += 1
            except Exception as e:
                if not ignore_errors:
                    raise
                print(f"[warn] Failed to read {fp}: {e}")
    return total_lines, written_lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge JSONL files in a directory into one JSONL")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing JSONL files")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Output JSONL file path")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--ignore-errors", action="store_true", help="Continue on read errors (log and skip)")

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_jsonl: Path = args.out_jsonl
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found or not a directory: {input_dir}")
        return 1

    files = collect_jsonl_files(input_dir, "*.jsonl", args.recursive)
    # Exclude output file if it's inside the input directory or matches pattern
    files = [p for p in files if p.resolve() != out_jsonl.resolve()]
    if not files:
        print(f"No files matched pattern '\"*.jsonl\"' under {input_dir}")
        return 1

    files = sort_files(files)

    total_lines, written_lines = merge_jsonl(
        files,
        out_jsonl,
        skip_empty=True,
        ignore_errors=args.ignore_errors,
    )

    print(f"Merged {len(files)} files -> {out_jsonl}")
    print(f"Lines read: {total_lines}, lines written: {written_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
