"""
Prepare pretraining corpus for Part 2 BERT and GPT models.

Reads the raw Hindi-Marathi parallel corpus (and the already-processed
Part 1 .src/.tgt files) and creates a flat text file where each line
is one sentence.  Both Hindi and Marathi sentences are included so
the pretrained models learn representations for both languages.

Output: data/pretrain_corpus.txt
"""

from pathlib import Path

DATA_DIR = Path("../part1nmt/data")

SOURCES = [
    # Monolingual sentences from the parallel corpus
    DATA_DIR / "train.hi",
    DATA_DIR / "train.mr",
    DATA_DIR / "valid.hi",
    DATA_DIR / "valid.mr",
    DATA_DIR / "test.hi",
    DATA_DIR / "test.mr",
]

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "pretrain_corpus.txt"

lines_written = 0
with open(out_path, "w", encoding="utf-8") as out_f:
    for src_path in SOURCES:
        if not src_path.exists():
            print(f"  WARNING: {src_path} not found — skipping")
            continue
        with open(src_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out_f.write(line + "\n")
                    lines_written += 1

print(f"Pretraining corpus created: {out_path}")
print(f"Total lines: {lines_written:,}")
