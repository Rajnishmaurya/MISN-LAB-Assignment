from sklearn.model_selection import train_test_split

train_hi = open("data/train.hi", encoding="utf-8").read().splitlines()
train_mr = open("data/train.mr", encoding="utf-8").read().splitlines()

train_hi_new, valid_hi, train_mr_new, valid_mr = train_test_split(
    train_hi,
    train_mr,
    test_size=0.1,
    random_state=42
)

with open("data/train_new.hi", "w", encoding="utf-8") as f:
    f.write("\n".join(train_hi_new))

with open("data/train_new.mr", "w", encoding="utf-8") as f:
    f.write("\n".join(train_mr_new))

with open("data/valid.hi", "w", encoding="utf-8") as f:
    f.write("\n".join(valid_hi))

with open("data/valid.mr", "w", encoding="utf-8") as f:
    f.write("\n".join(valid_mr))

print("Validation split created.")