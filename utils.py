def get_data(indices, data):
    queries = []
    passages = []

    for i in indices:
        query = data[i]["question"]
        queries.append(query)
    
        passage = data[i]["context"]
        passages.append(passage)
    
    return queries, passages

def get_batches(queries, passages, batch_size):
    batches = []
    pending = []
    current_batch = {"question": [], "context": []}
    current_passages = set()
    current_questions = set()

    def process_pending():
        nonlocal current_batch, current_passages, current_questions, pending
        new_pending = []
        for q, p in pending:
            if (p not in current_passages and 
                q not in current_questions and 
                len(current_batch["question"]) < batch_size):
                current_batch["question"].append(q)
                current_batch["context"].append(p)
                current_passages.add(p)
                current_questions.add(q)
            else:
                new_pending.append((q, p))
        pending = new_pending

    for q, p in zip(queries, passages):
        if p in current_passages or q in current_questions:
            pending.append((q, p))
        else:
            current_batch["question"].append(q)
            current_batch["context"].append(p)
            current_passages.add(p)
            current_questions.add(q)

        if len(current_batch["question"]) == batch_size:
            batches.append(current_batch)
            current_batch = {"question": [], "context": []}
            current_passages = set()
            current_questions = set()
            process_pending()

    while pending and len(current_batch["question"]) < batch_size:
        q, p = pending.pop(0)
        if p not in current_passages and q not in current_questions:
            current_batch["question"].append(q)
            current_batch["context"].append(p)
            current_passages.add(p)
            current_questions.add(q)
        else:
            pending.append((q, p))
            if all(p in current_passages or q in current_questions for q, p in pending):
                break

    if current_batch["question"]:
        batches.append(current_batch)

    return batches