import torch
import numpy as np

def get_data(data, use_hard_negatives=True):
    queries = []
    passages = []
    hard_negatives = []

    for i in range(len(data)):
        queries.append(data[i]["query"])
        passages.append(data[i]["positive"])
        if use_hard_negatives:
            hard_negatives.append(data[i]["hard_negatives"])

    return queries, passages, hard_negatives

def get_batches(queries, passages, hard_negatives, batch_size, use_hard_negatives=True):
    batches = []
    pending = []
    current_batch = {"question": [], "context": []}
    if use_hard_negatives:
        current_batch["hard_negatives"] = []

    current_passages = set()
    current_questions = set()

    def process_pending():
        nonlocal current_batch, current_passages, current_questions, pending
        new_pending = []
        for item in pending:
            q, p = item[0], item[1]
            hn = item[2] if use_hard_negatives else None

            if (p not in current_passages and 
                q not in current_questions and 
                len(current_batch["question"]) < batch_size):
                current_batch["question"].append(q)
                current_batch["context"].append(p)
                if use_hard_negatives:
                    current_batch["hard_negatives"].append(hn)
                current_passages.add(p)
                current_questions.add(q)
            else:
                new_pending.append(item)
        pending = new_pending

    if use_hard_negatives:
        iterator = zip(queries, passages, hard_negatives)
    else:
        iterator = zip(queries, passages)

    for item in iterator:
        q, p = item[0], item[1]
        hn = item[2] if use_hard_negatives else None

        if p in current_passages or q in current_questions:
            pending.append(item)
        else:
            current_batch["question"].append(q)
            current_batch["context"].append(p)
            if use_hard_negatives:
                current_batch["hard_negatives"].append(hn)
            current_passages.add(p)
            current_questions.add(q)

        if len(current_batch["question"]) == batch_size:
            batches.append(current_batch)
            current_batch = {"question": [], "context": []}
            if use_hard_negatives:
                current_batch["hard_negatives"] = []
            current_passages = set()
            current_questions = set()
            process_pending()

    while pending and len(current_batch["question"]) < batch_size:
        item = pending.pop(0)
        q, p = item[0], item[1]
        hn = item[2] if use_hard_negatives else None

        if p not in current_passages and q not in current_questions:
            current_batch["question"].append(q)
            current_batch["context"].append(p)
            if use_hard_negatives:
                current_batch["hard_negatives"].append(hn)
            current_passages.add(p)
            current_questions.add(q)
        else:
            pending.append(item)
            if all(item[1] in current_passages or item[0] in current_questions for item in pending):
                break

    if current_batch["question"]:
        batches.append(current_batch)

    return batches

def get_ndcg(scores, labels, k):
    top_indices = torch.topk(scores, k=k, largest=True).indices
    sorted_labels = labels[top_indices]

    dcg = sum((sorted_labels[i].item() / np.log2(i + 2)) for i in range(len(sorted_labels)))
    idcg = sum((1 / np.log2(i + 2)) for i in range(min(int(labels.sum().item()), k)))

    return dcg / idcg if idcg > 0 else 0.0