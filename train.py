import json
from utils import get_data, get_batches
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
import random
from datasets import load_dataset
from model import HardNegativeInfoNCELoss, LitContrastiveModel, ContrastiveLoss
from lightning import Trainer
from lightning import seed_everything
import argparse

class QPDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "index": s["index"],
            "question": s["query"],
            "context": s["positive"],
            "hard_negatives": s.get("hard_negatives"),
        }


def test_collate_fn(batch):
    return {
        "index":    [b["index"] for b in batch],
        "question": [b["question"] for b in batch],
        "context":  [b["context"] for b in batch],
    }


def make_unique_test_split(val_data_path, test_fraction=0.1, seed=42):
    base_dataset = QPDataset(val_data_path)
    n = len(base_dataset)
    desired_test_size = int(n * test_fraction)

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    seen_q = set()
    seen_p = set()
    test_idx = []
    train_idx = []

    for idx in indices:
        sample = base_dataset[idx]
        q = sample["question"]
        p = sample["context"]

        if (q not in seen_q) and (p not in seen_p) and (len(test_idx) < desired_test_size):
            test_idx.append(idx)
            seen_q.add(q)
            seen_p.add(p)
        else:
            train_idx.append(idx)

    base_train_ds = Subset(base_dataset, train_idx)
    base_test_ds = Subset(base_dataset, test_idx)

    print(f"TEST DATASET SIZE = {len(base_test_ds)} (unique questions & contexts)")
    print(f"VAL-TRAIN PART FROM VAL FILE = {len(base_train_ds)}")

    return base_train_ds, base_test_ds


def to_list_indices(x):
    if isinstance(x, list):
        return x
    return [x]

def load_train_data_filtered(train_data_path, test_global_indices):
    train_data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            idx_field = sample["index"]
            indices = to_list_indices(idx_field)

            if any(i in test_global_indices for i in indices):
                continue

            train_data.append(sample)

    print(f"Full train file size (raw): {sum(1 for _ in open(train_data_path, 'r', encoding='utf-8'))}")
    print(f"Train examples after removing test indices: {len(train_data)}")

    return train_data


def start(train_data_path, val_data_path, batch_size, model_path, loss_type, train_mode):

    seed_everything(42, workers=True)

    base_train_ds, base_test_ds = make_unique_test_split(val_data_path, test_fraction=0.1, seed=42)
    testloader = DataLoader(base_test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_collate_fn)

    test_global_indices = set()
    for sample in base_test_ds:
        idx_field = sample["index"]
        for idx in to_list_indices(idx_field):
            test_global_indices.add(idx)

    print(f"UNIQUE TEST INDICES COUNT = {len(test_global_indices)}")

    seed_everything(42, workers=True)

    train_data_filtered = load_train_data_filtered(train_data_path, test_global_indices)

    indexes, queries, passages, hard_negatives = get_data(train_data_filtered)
    print(f"Train examples going into batching: {len(queries)}")

    batches = get_batches(indexes, queries, passages, hard_negatives, batch_size)
    print(f"TRAIN BATCHES AFTER get_batches = {len(batches)}")

    filtered_len = len(batches)
    train_size = int(0.99 * filtered_len)
    val_size   = filtered_len - train_size

    train_dataset, val_dataset = random_split(batches, [train_size, val_size])

    print(f"FINAL TRAIN BATCHES = {len(train_dataset)}")
    print(f"FINAL VAL   BATCHES = {len(val_dataset)}")

    trainloader = DataLoader(train_dataset, batch_size=None, collate_fn=lambda x: x, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=None, collate_fn=lambda x: x, shuffle=False)

    if loss_type == "HardNegative":
        loss_fn = HardNegativeInfoNCELoss()
        print("FUNCTION LOSS: HardNegative")
    elif loss_type == "Contrastive":
        loss_fn = ContrastiveLoss()
        print("FUNCTION LOSS: ContrastiveLoss")

    model = LitContrastiveModel(
        model_path=model_path,
        loss_fn=loss_fn,
        lr=9e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        epochs=2,
        train_len=len(trainloader),
        max_length=512,
        query_prefix = "search_query: ",
        doc_prefix = "search_document: ",
        pooling_method = "mean"
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        deterministic=True,
    )

    if train_mode == "True":
        trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    
    trainer.test(model, dataloaders=testloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--loss_type", type=str, default="HardNegative")
    parser.add_argument("--train_mode", type=str, default="True")
    args = parser.parse_args()

    start(args.train_data_path, args.val_data_path, args.batch_size, args.model_path, args.loss_type, args.train_mode)