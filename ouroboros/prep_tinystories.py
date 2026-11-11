"""Turn a TinyStories text file into an ASCII character corpus for train_tm.

Keeps printable ASCII and newlines; maps curly quotes and dashes to ASCII; drops every other
character; replaces the story separator <|endoftext|> with a blank line.  Usage:
  python -m ouroboros.prep_tinystories <in.txt> <out.txt> [--max_mb 50]"""
from __future__ import annotations

import argparse
import string

MAP = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "}
KEEP = set(string.printable) - set("\t\r\x0b\x0c")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--max_mb", type=float, default=50.0)
    a = p.parse_args()
    raw = open(a.src, encoding="utf-8", errors="ignore").read(int(a.max_mb * 2**20))
    raw = raw[: raw.rfind("<|endoftext|>")] if "<|endoftext|>" in raw else raw  # end on a whole story
    raw = raw.replace("<|endoftext|>", "\n")
    out = "".join(MAP.get(c, c) for c in raw)
    out = "".join(c for c in out if c in KEEP)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    open(a.dst, "w", encoding="utf-8").write(out)
    print(f"{len(raw):,} chars in, {len(out):,} out, {len(set(out))} distinct characters")


if __name__ == "__main__":
    main()
