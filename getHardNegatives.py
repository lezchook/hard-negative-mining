import json
import torch
from torch import Tensor
import torch.nn.functional as F
import argparse
from tqdm.auto import tqdm


def hard_negative_mining_batch(queries_emb: Tensor, contexts_emb: Tensor, pos_idx: Tensor, contexts: list[str], margin: float = 0.97, top_k: int = 5) -> list[list[int]]:
    queries_emb = F.normalize(queries_emb, p=2, dim=1)
    contexts_emb = F.normalize(contexts_emb, p=2, dim=1)

    all_scores = queries_emb @ contexts_emb.T
    all_scores = (all_scores + 1.0) / 2.0

    device = queries_emb.device
    B = queries_emb.size(0)

    row_idx = torch.arange(B, device=device)

    pos_scores = all_scores[row_idx, pos_idx]

    thresholds = pos_scores * margin

    mask = torch.ones_like(all_scores, dtype=torch.bool)
    mask[row_idx, pos_idx] = False

    threshold_mask = all_scores < thresholds.unsqueeze(1)
    final_mask = mask & threshold_mask

    hard_negatives_indices: list[list[int]] = []

    for i in range(B):
        valid_indices = torch.where(final_mask[i])[0]

        if len(valid_indices) == 0:
            hard_negatives_indices.append([])
            continue

        valid_scores = all_scores[i][valid_indices]
        sorted_scores, sorted_idx = torch.sort(valid_scores, descending=True)
        sorted_candidate_indices = valid_indices[sorted_idx]

        pos_global_idx = pos_idx[i].item()
        positive_text = contexts[pos_global_idx]

        seen_texts = {positive_text}
        selected_indices: list[int] = []

        for ctx_idx in sorted_candidate_indices.tolist():
            ctx_text = contexts[ctx_idx]

            if ctx_text in seen_texts:
                continue

            seen_texts.add(ctx_text)
            selected_indices.append(ctx_idx)

            if len(selected_indices) == top_k:
                break

        hard_negatives_indices.append(selected_indices)

    return hard_negatives_indices


def get_data_from_json(data_path: str):
    queries = []
    contexts = []
    queries_emb = []
    contexts_emb = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Чтение файла"):
            sample = json.loads(line)
            queries.append(sample["query"])
            contexts.append(sample["context"])
            queries_emb.append(sample["query_emb"])
            contexts_emb.append(sample["context_emb"])

    queries_emb = torch.tensor(queries_emb, dtype=torch.float32)
    contexts_emb = torch.tensor(contexts_emb, dtype=torch.float32)

    return queries, contexts, queries_emb, contexts_emb


def start_mining(data_path, output_file, margin=0.97, top_k=5, batch_size=None):
    queries, contexts, queries_emb, contexts_emb = get_data_from_json(data_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries_emb = queries_emb.to(device)
    contexts_emb = contexts_emb.to(device)

    N = len(queries)

    with torch.no_grad():
        if batch_size is None:
            pos_idx = torch.arange(N, device=device)
            hard_negatives_indices = hard_negative_mining_batch(queries_emb, contexts_emb, pos_idx, contexts, margin, top_k)
        else:
            hard_negatives_indices: list[list[int]] = []
            num_batches = (N + batch_size - 1) // batch_size

            for b in tqdm(range(num_batches), desc="Майнинг батчей"):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, N)

                batch_queries = queries_emb[start_idx:end_idx]  
            
                batch_pos_idx = torch.arange(start_idx, end_idx, device=device)
                batch_hn = hard_negative_mining_batch(batch_queries, contexts_emb, batch_pos_idx, contexts, margin, top_k)
                hard_negatives_indices.extend(batch_hn)

    kept = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx in tqdm(range(N), desc="Запись результата"):
            hn_indices = hard_negatives_indices[idx]

            if len(hn_indices) < top_k:
                skipped += 1
                continue

            hn_docs = [contexts[i] for i in hn_indices]

            result = {
                "index": idx,
                "query": queries[idx],
                "positive": contexts[idx],
                "hard_negatives": hn_docs,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            kept += 1


    print(f"Всего примеров: {N}")
    print(f"Сохранено примеров: {kept}")
    print(f"Пропущено (меньше {top_k} уникальных негативов): {skipped}")
    print(f"Результаты сохранены в {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--margin", type=float, default=0.97)
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