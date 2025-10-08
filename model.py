import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from utils import get_ndcg
import numpy as np

def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]

class HardNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.02, false_negative_margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.false_negative_margin = false_negative_margin

    def forward(self, query, passage, negative_passages):
        batch_size = query.size(0)

        s_positive = F.cosine_similarity(query, passage, dim=-1) # s(q_i, d+_i)
        s_hard_neg = F.cosine_similarity(query.unsqueeze(1), negative_passages, dim=-1) # s(q_i, d-_i,k)
        
        s_in_batch_docs = torch.mm(query, passage.t()) / (query.norm(dim=1, keepdim=True) @ passage.norm(dim=1, keepdim=True).t() + 1e-8)
        s_in_batch_queries = torch.mm(query, query.t()) / (query.norm(dim=1, keepdim=True) @ query.norm(dim=1, keepdim=True).t() + 1e-8)
        
        threshold = s_positive.unsqueeze(1) + self.false_negative_margin

        mask_hard = (s_hard_neg <= threshold).float()
        mask_docs = (s_in_batch_docs <= threshold).float()
        mask_docs = mask_docs * (1 - torch.eye(batch_size, device=query.device))
        
        mask_queries = (s_in_batch_queries <= threshold).float()
        mask_queries = mask_queries * (1 - torch.eye(batch_size, device=query.device))

        s_positive_scaled = s_positive / self.temperature
        s_hard_neg_scaled = s_hard_neg / self.temperature
        s_in_batch_docs_scaled = s_in_batch_docs / self.temperature
        s_in_batch_queries_scaled = s_in_batch_queries / self.temperature
        
        
        exp_positive = torch.exp(s_positive_scaled)
        
        exp_hard_neg = torch.exp(s_hard_neg_scaled) * mask_hard
        sum_hard_neg = exp_hard_neg.sum(dim=1)
        
        exp_docs = torch.exp(s_in_batch_docs_scaled) * mask_docs
        sum_docs = exp_docs.sum(dim=1)
        
        exp_queries = torch.exp(s_in_batch_queries_scaled) * mask_queries
        sum_queries = exp_queries.sum(dim=1)
        
        Z = exp_positive + sum_hard_neg + sum_docs + sum_queries
   
        loss = -torch.log(exp_positive / (Z + 1e-8))
        
        return loss.mean()
    
class LitContrastiveModel(LightningModule):
    def __init__(self, model_path, loss_fn, lr, weight_decay, warmup_ratio, epochs, train_len):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.train_len = train_len

    def forward(self, x):
        query = self.tokenizer(x["question"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        passage = self.tokenizer(x["context"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_output = self.model(**query)
        passage_output = self.model(**passage)
        query_emb = pool(query_output.last_hidden_state, query["attention_mask"], pooling_method="mean")
        passage_emb = pool(passage_output.last_hidden_state, passage["attention_mask"],pooling_method="mean")

        hard_negatives = []
        for i in range(len(x["hard_negatives"])):
            tokens = self.tokenizer(x["hard_negatives"][i], return_tensors="pt", truncation=True, padding=True).to(self.device)
            output = self.model(**tokens)
            output_emb = pool(output.last_hidden_state, tokens["attention_mask"], pooling_method="mean")
            hard_negatives.append(output_emb)

        hard_negatives_emb = torch.stack(hard_negatives)

        return query_emb, passage_emb, hard_negatives_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.train_len * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def training_step(self, batch, batch_idx):
        query_emb, passage_emb, hard_negatives_emb = self(batch)
        loss = self.loss_fn(query_emb, passage_emb, hard_negatives_emb)
        self.log("train_loss", loss, batch_size=query_emb.shape[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            print(f"Training Loss: {train_loss:.4f}")

    def validation_step(self, x, batch_idx):
        query = self.tokenizer(x["question"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        passage = self.tokenizer(x["context"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_output = self.model(**query)
        passage_output = self.model(**passage)
        query_emb = pool(query_output.last_hidden_state, query["attention_mask"], pooling_method="mean")
        passage_emb = pool(passage_output.last_hidden_state, passage["attention_mask"],pooling_method="mean")
        
        scores = torch.zeros(len(query_emb), len(passage_emb))
        for i in range(len(query_emb)):
            scores[i] = F.cosine_similarity(query_emb[i].unsqueeze(0), passage_emb)

        ndcg_scores = []
        for i in range(len(x["question"])):
            labels = torch.zeros(len(x["context"]))
            labels[i] = 1
            ndcg_scores.append(get_ndcg(scores[i], labels, k=2))
        
        ndcg_scores = np.array(ndcg_scores)
        ndcg_scores = torch.tensor(ndcg_scores, device=self.device)

        self.log("val_ndcg@2", ndcg_scores.mean(), batch_size=len(x["question"]), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return ndcg_scores.mean()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_ndcg@2")
        if val_loss is not None:
            print(f"Validation NDCG@2: {val_loss:.4f}")

    def test_step(self, x, batch_idx):
        query = self.tokenizer(x["question"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        passage = self.tokenizer(x["context"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_output = self.model(**query)
        passage_output = self.model(**passage)
        query_emb = pool(query_output.last_hidden_state, query["attention_mask"], pooling_method="mean")
        passage_emb = pool(passage_output.last_hidden_state, passage["attention_mask"],pooling_method="mean")
        
        scores = torch.zeros(len(query_emb), len(passage_emb))
        for i in range(len(query_emb)):
            scores[i] = F.cosine_similarity(query_emb[i].unsqueeze(0), passage_emb)

        ndcg_scores = []
        for i in range(len(x["question"])):
            labels = torch.zeros(len(x["context"]))
            labels[i] = 1
            ndcg_scores.append(get_ndcg(scores[i], labels, k=2))
        
        ndcg_scores = np.array(ndcg_scores)
        ndcg_scores = torch.tensor(ndcg_scores, device=self.device)

        self.log("test_ndcg@2", ndcg_scores.mean(), batch_size=len(x["question"]), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return ndcg_scores.mean()
    
    def on_test_epoch_end(self):
        test_ndcg = self.trainer.callback_metrics.get("test_ndcg@2")
        if test_ndcg is not None:
            print(f"Test NDCG@2: {test_ndcg:.4f}")