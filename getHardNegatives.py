import json
import torch
from torch import Tensor
import argparse
from tqdm.auto import tqdm

def hard_negative_mining_batch(queries_emb: Tensor, contexts_emb: Tensor, margin: float = 0.95, top_k: int = 5) -> list[list[int]]:
    all_scores = queries_emb @ contexts_emb.T
    
    # Позитивные scores - диагональ матрицы
    pos_scores = torch.diag(all_scores)
    
    # Threshold для каждого query
    thresholds = pos_scores * margin
    
    mask = torch.ones_like(all_scores, dtype=torch.bool)
    mask.fill_diagonal_(False)
    
    threshold_mask = all_scores < thresholds.unsqueeze(1)
    final_mask = mask & threshold_mask
    
    # Извлекаем top-k hard negatives для каждого query
    hard_negatives_indices = []
    for i in range(len(queries_emb)):
        valid_indices = torch.where(final_mask[i])[0]
        
        if len(valid_indices) > 0:
            valid_scores = all_scores[i][valid_indices]
        
            sorted_scores, sorted_idx = torch.sort(valid_scores, descending=True)
            k = min(top_k, len(sorted_idx))
            top_indices = valid_indices[sorted_idx[:k]].tolist()
        else:
            top_indices = []
        
        hard_negatives_indices.append(top_indices)
    
    return hard_negatives_indices


def get_data_from_json(data_path: str) -> tuple:
    queries = []
    contexts = []
    queries_emb = []
    contexts_emb = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Чтение файла"):
            sample = json.loads(line)
            queries.append(sample["query"])
            contexts.append(sample["context"])
            queries_emb.append(sample["query_emb"][0])
            contexts_emb.append(sample["context_emb"][0])
    
    queries_emb = torch.tensor(queries_emb)
    contexts_emb = torch.tensor(contexts_emb)
    
    return queries, contexts, queries_emb, contexts_emb


def start_mining(data_path: str, output_file: str, margin: float = 0.95, top_k: int = 5, batch_size: int = None):
    queries, contexts, queries_emb, contexts_emb = get_data_from_json(data_path)
    
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    
    queries_emb = queries_emb.to(device)
    contexts_emb = contexts_emb.to(device)
    
    if batch_size is None:
        hard_negatives_indices = hard_negative_mining_batch(queries_emb, contexts_emb, margin, top_k)
    else:
        hard_negatives_indices = []
        num_batches = (len(queries) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            
            batch_hn = hard_negative_mining_batch(queries_emb[start_idx:end_idx], contexts_emb, margin, top_k)
            hard_negatives_indices.extend(batch_hn)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in tqdm(range(len(queries))):
            hn_indices = hard_negatives_indices[idx]
            hn_docs = [contexts[i] for i in hn_indices]
            
            result = {
                "index": idx,
                "query": queries[idx],
                "positive": contexts[idx],
                "hard_negatives": hn_docs
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Результаты сохранены в {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--margin", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=None)
    
    args = parser.parse_args()
    
    start_mining(
        args.data_path,
        args.output_file,
        margin=args.margin,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )