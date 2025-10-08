import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn.functional as F
import argparse
from tqdm.auto import tqdm
import json
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def encode_texts(model: AutoModel, tokenizer: AutoTokenizer, input_text: list[str], device: str = "cuda") -> Tensor:
    texts_dict = tokenizer(input_text, max_length=4096, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**texts_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, texts_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def start_inference(data_path, output_file, batch_size):
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = load_dataset(data_path)
    all_queries = [get_detailed_instruct(task, sample["question"]) for sample in data["train"]]
    all_contexts = [sample["context"] for sample in data["train"]]
    
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct").to(device)
    model.eval()
    
    with open(output_file, "a", encoding="utf-8") as f:
        pbar = tqdm(total=len(data["train"]), desc="Processing", unit="item")

        for start_idx in range(0, len(data["train"]), batch_size):
            end_idx = min(start_idx + batch_size, len(data["train"]))
            batch_queries = all_queries[start_idx:end_idx]
            batch_contexts = all_contexts[start_idx:end_idx]

            batch_texts = batch_queries + batch_contexts
            embeddings = encode_texts(model, tokenizer, batch_texts, device)

            query_embs = embeddings[:len(batch_queries)]
            context_embs = embeddings[len(batch_queries):]

            for i, (q, c, q_emb, c_emb) in enumerate(zip(batch_queries, batch_contexts, query_embs, context_embs)):
                idx = start_idx + i
                result = {
                    "index": idx,
                    "query": q,
                    "context": c,
                    "query_emb": q_emb.cpu().tolist(),
                    "context_emb": c_emb.cpu().tolist()
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