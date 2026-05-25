from pathlib import Path

data_dir = Path("data")

train_hi = open(data_dir / "train.hi", encoding="utf-8").read().splitlines()
train_mr = open(data_dir / "train.mr", encoding="utf-8").read().splitlines()

valid_hi = open(data_dir / "valid.hi", encoding="utf-8").read().splitlines()
valid_mr = open(data_dir / "valid.mr", encoding="utf-8").read().splitlines()

test_hi = open(data_dir / "test.hi", encoding="utf-8").read().splitlines()
test_mr = open(data_dir / "test.mr", encoding="utf-8").read().splitlines()


def build_bidirectional(src_hi, src_mr):

    src_lines = []
    tgt_lines = []

    for hi, mr in zip(src_hi, src_mr):

        hi = hi.strip()
        mr = mr.strip()

        # Hindi -> Marathi
        src_lines.append(f"<hi2mr> {hi}")
        tgt_lines.append(mr)

        # Marathi -> Hindi
        src_lines.append(f"<mr2hi> {mr}")
        tgt_lines.append(hi)

    return src_lines, tgt_lines


for split_name, hi_data, mr_data in [
    ("train", train_hi, train_mr),
    ("valid", valid_hi, valid_mr),
    ("test", test_hi, test_mr),
]:

    src, tgt = build_bidirectional(hi_data, mr_data)

    with open(data_dir / f"{split_name}.src", "w", encoding="utf-8") as f:
        f.write("\n".join(src))

    with open(data_dir / f"{split_name}.tgt", "w", encoding="utf-8") as f:
        f.write("\n".join(tgt))

print("Bidirectional multilingual dataset created.")
