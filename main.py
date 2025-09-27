import json
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from utils import get_data, get_batches
import argparse
from tqdm.auto import tqdm
import os

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def hard_negative_mining(batch_index: int,  query: str, positive_doc: str, q_emb: Tensor, pos_emb: Tensor, cand_embs: Tensor, candidate_docs: list, 
                            margin: float = 0.95, top_k: int = 5, device: str ="cuda") -> dict:
    pos_score = (q_emb @ pos_emb.T).item()
    scores = (q_emb @ cand_embs.T).squeeze(0)

    threshold = pos_score * margin
    mask = scores < threshold
    filtered_scores = scores[mask]
    filtered_indices = torch.arange(len(candidate_docs), device=device)[mask]

    # top-k самых трудных
    if len(filtered_scores) > 0:
        topk = torch.topk(filtered_scores, min(top_k, len(filtered_scores)))[1]
        hard_negatives = [candidate_docs[idx] for idx in filtered_indices[topk]]
    else:
        hard_negatives = []

    return {
        "batch_index": batch_index,
        "query": query,
        "positive": positive_doc,
        "hard_negatives": hard_negatives
    }

def get_batched_data(data_path: str, batch_size: int) -> list:
    dataset = load_dataset(data_path)
    train_data = [{"question": q, "context": c} for q, c in zip(dataset["train"]["question"], dataset["train"]["context"]) if c is not None]
    queries_train, passages_train = get_data(range(len(train_data)), train_data)
    data_batched = get_batches(queries_train, passages_train, batch_size)
    return data_batched

def encode_texts(model: AutoModel, tokenizer: AutoTokenizer, input_texts: list, device: str = "cuda") -> Tensor:
    texts_dict = tokenizer(input_texts, max_length=4096, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**texts_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, texts_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def start_mining(data_path, batch_size, output_file):
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = get_batched_data(data_path, batch_size)   
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    model = AutoModel.from_pretrained("intfloat/e5-mistral-7b-instruct").to(device)
    model.eval()
    
    with open(output_file, "a", encoding="utf-8") as f:
        pbar = tqdm(total=len(data), desc="Processing", unit="item")
        for batch_index, batch in enumerate(data):
            all_contexts = batch["context"]
            ctx_embs = encode_texts(model, tokenizer, all_contexts, device)

            query_text = [get_detailed_instruct(task, q) for q in batch["question"]]
            q_embs = encode_texts(model, tokenizer, query_text, device)
            for idx in range(batch_size):
                q_emb = q_embs[idx:idx+1]
                pos_emb = ctx_embs[idx:idx+1]
                cand_embs = torch.cat([ctx_embs[:idx], ctx_embs[idx+1:]], dim=0)

                query = batch["question"][idx]
                positive_doc = batch["context"][idx]
                candidate_docs = batch["context"][0:idx] + batch["context"][idx + 1:batch_size]
                
                result = hard_negative_mining(batch_index, query, positive_doc, q_emb, pos_emb, cand_embs, candidate_docs, margin=0.95, top_k=5, device=device)

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            f.flush()
            pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    start_mining(args.data_path, args.batch_size, args.output_file)