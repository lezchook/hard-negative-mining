import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

def set_seed(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def start_inference(data_path: str, output_file: str):
    dataset = load_dataset(data_path, 'all-processed')
    train_data = dataset["train"]

    all_queries = [sample["input"] for sample in train_data]
    all_contexts = [sample["output"] for sample in train_data]

    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=len(all_queries), desc="Processing", unit="item")

        for idx in range(len(all_queries)):
            result = {
                "index": idx,
                "query": all_queries[idx],
                "positive": all_contexts[idx],
                "hard_negatives": [],
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            f.flush()
            pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    start_inference(args.data_path, args.output_file)