import json
import torch
from torch import Tensor
import argparse
from tqdm.auto import tqdm

def hard_negative_mining(index: int,  query: str, positive_doc: str, candidate_docs: list[str], 
                            q_emb: Tensor, pos_emb: Tensor, cand_embs: Tensor, margin: float = 0.95, top_k: int = 5) -> dict:
    pos_score = (q_emb @ pos_emb).item()
    scores = (q_emb @ cand_embs.T).squeeze(0)

    threshold = pos_score * margin
    mask = scores < threshold
    filtered_scores = scores[mask]
    filtered_indices = torch.arange(len(candidate_docs))[mask]

    # top-k самых трудных
    hard_negatives = []
    if len(filtered_scores) > 0:
        sorted_scores, sorted_idx = torch.sort(filtered_scores, descending=True)
        for idx in sorted_idx:
            doc = candidate_docs[filtered_indices[idx]]
            if doc not in hard_negatives:
                hard_negatives.append(doc)
            if len(hard_negatives) >= top_k:
                break
    
    return {
        "index": index,
        "query": query,
        "positive": positive_doc,
        "hard_negatives": hard_negatives
    }

def get_data_from_json(data_path: str) -> list:
    queries = []
    contexts = []
    queries_emb = []
    contexts_emb = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            queries.append(sample["query"])
            contexts.append(sample["context"])
            queries_emb.append(sample["query_emb"][0])
            contexts_emb.append(sample["context_emb"][0])

    queries_emb = torch.tensor(queries_emb)
    contexts_emb = torch.tensor(contexts_emb)        
    length = len(queries)
    return queries, contexts, queries_emb, contexts_emb, length

def start_mining(data_path, output_file):
    queries, contexts, queries_emb, contexts_emb, length = get_data_from_json(data_path)   

    with open(output_file, "a", encoding="utf-8") as f:
        pbar = tqdm(total=length, desc="Processing", unit="item")
        for idx in range(length):
            query = queries[idx]
            positive_doc = contexts[idx]
            query_emb = queries_emb[idx]
            positive_doc_emb = contexts_emb[idx]

            candidate_docs = contexts[0:idx] + contexts[idx + 1:length]
            candidate_docs_emb = torch.cat((contexts_emb[0:idx], contexts_emb[idx + 1:length]), dim=0)

            result = hard_negative_mining(idx, query, positive_doc, candidate_docs, query_emb, positive_doc_emb, candidate_docs_emb, margin=0.95, top_k=5)
            
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    start_mining(args.data_path, args.output_file)