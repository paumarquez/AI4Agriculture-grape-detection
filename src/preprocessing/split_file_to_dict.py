import json

train_file_name = 'data/ignasi/split1/train.json'
val_file_name = 'data/ignasi/split1/val.json'
out_name = 'data/splits/split4.json'

with open(train_file_name, 'r') as fd:
    train = json.load(fd)
train_ids = [d["filename"][:-4] for d in train]

with open(val_file_name, 'r') as fd:
    train = json.load(fd)
val_ids = [d["filename"][:-4] for d in train]

with open(out_name, 'w') as fd:
    json.dump({
    "train": train_ids,
    "val": val_ids
    }, fd)
