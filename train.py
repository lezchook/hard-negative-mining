import json
from utils import get_data, get_batches
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import HardNegativeInfoNCELoss, LitContrastiveModel
from lightning import Trainer
from lightning import seed_everything
import argparse

def get_trainloader(data_path, batch_size):
    train_data = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            train_data.append(sample)

    queries, passages, hard_negatives = get_data(train_data)
    batches = get_batches(queries, passages, hard_negatives, batch_size)
    trainloader = DataLoader(batches, batch_size=None, collate_fn=lambda x: x, shuffle=True)
    
    return trainloader

def get_valloader(data_path, batch_size):
    data = load_dataset(data_path)
    val_data = [{"query": q, "positive": c} for q, c in zip(data["validation"]["question"], data["validation"]["context"]) if c is not None]
    queries, passages, _ = get_data(val_data, use_hard_negatives=False)
    val_data_batched = get_batches(queries=queries, passages=passages, hard_negatives=None, batch_size=batch_size, use_hard_negatives=False)
    valloader = DataLoader(val_data_batched, batch_size=None, collate_fn=lambda x: x, shuffle=False)

    return valloader

def get_testloader(data_path, batch_size):
    data = load_dataset(data_path)
    test_data = [{"query": q, "positive": c} for q, c in zip(data["test"]["question"], data["test"]["context"]) if c is not None]
    queries, passages, _ = get_data(test_data, use_hard_negatives=False)
    test_data_batched = get_batches(queries=queries, passages=passages, hard_negatives=None, batch_size=batch_size, use_hard_negatives=False)
    testloader = DataLoader(test_data_batched, batch_size=None, collate_fn=lambda x: x, shuffle=False)

    return testloader

def start(train_data_path, val_data_path, batch_size):
    seed_everything(42, workers=True)

    trainloader = get_trainloader(train_data_path, batch_size)
    valloader = get_valloader(val_data_path, batch_size)
    testloader = get_testloader(val_data_path, batch_size)

    model = LitContrastiveModel(
        model_path="cointegrated/rubert-tiny2",
        loss_fn=HardNegativeInfoNCELoss(),
        lr=9e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        epochs=5,
        train_len=len(trainloader)
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=5,
        deterministic=True
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model, dataloaders=testloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=str, required=True)
    args = parser.parse_args()

    start(args.train_data_path, args.val_data_path, args.batch_size)