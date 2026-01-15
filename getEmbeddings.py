import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

def set_seed(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"

def encode_texts(model: AutoModel, tokenizer: AutoTokenizer, input_texts: list[str], device: str = "cuda", max_length: int = 512) -> Tensor:
    batch = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**batch)

    embeddings = average_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def start_inference(data_path: str, output_file: str, batch_size: int):
    task = "Given a web search query, retrieve relevant passages that answer the query"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset(data_path, 'all-processed')
    train_data = dataset["train"]

    all_queries = [get_detailed_instruct(task, sample["input"]) for sample in train_data]
    all_queries_raw = [sample["input"] for sample in train_data]
    all_contexts = [sample["output"] for sample in train_data]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct", torch_dtype=torch.bfloat16).to(device)
    model.eval()

    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=len(train_data), desc="Processing", unit="item")

        for start_idx in range(0, len(train_data), batch_size):
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_queries = all_queries[start_idx:end_idx]
            batch_queries_raw = all_queries_raw[start_idx:end_idx]
            batch_contexts = all_contexts[start_idx:end_idx]

            batch_texts = batch_queries + batch_contexts
            embeddings = encode_texts(model, tokenizer, batch_texts, device=device)

            num_queries = len(batch_queries)
            query_embs = embeddings[:num_queries]
            context_embs = embeddings[num_queries:]

            for i, (q, c, q_emb, c_emb) in enumerate(zip(batch_queries_raw, batch_contexts, query_embs, context_embs)):
                idx = start_idx + i
                result = {
                    "index": idx,
                    "query": q,
                    "context": c,
                    "query_emb": q_emb.cpu().tolist(),
                    "context_emb": c_emb.cpu().tolist(),
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            f.flush()
            pbar.update(len(batch_queries))

        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    set_seed(42)
    start_inference(args.data_path, args.output_file, args.batch_size)