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

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, query, passage, negative_passages, temperature):
        s_positive = F.cosine_similarity(query, passage, dim=-1) / temperature
        s_negative = F.cosine_similarity(query.unsqueeze(1), negative_passages, dim=-1) / temperature

        exp_for_sum = torch.cat([s_positive.unsqueeze(-1), s_negative], dim=-1)
        log_exp_sum = torch.logsumexp(exp_for_sum, dim=-1)
        
        return (-s_positive + log_exp_sum).mean()

class HardNegativeInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.02, false_negative_margin: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.false_negative_margin = false_negative_margin
        self.eps = eps

    def forward(self, query, passage, negative_passages=None):
        device = query.device
        B = query.size(0)

        q = F.normalize(query, dim=-1)
        d_pos = F.normalize(passage, dim=-1)

        s_pos = (q * d_pos).sum(dim=-1)

        s_q_q = q @ q.t()
        s_q_d = q @ d_pos.t()
        s_d_d = d_pos @ d_pos.t()

        threshold = s_pos[:, None] + self.false_negative_margin
        eye = torch.eye(B, device=device, dtype=torch.bool)

        mask_qq = (~eye) & (s_q_q <= threshold)
        mask_qd = (~eye) & (s_q_d <= threshold)
        mask_dd = (~eye) & (s_d_d <= threshold)

        neg_inf = torch.finfo(q.dtype).min

        logits = [ (s_pos / self.temperature)[:, None] ]
        logits.append((s_q_q / self.temperature).masked_fill(~mask_qq, neg_inf).reshape(B, -1))
        logits.append((s_q_d / self.temperature).masked_fill(~mask_qd, neg_inf).reshape(B, -1))
        logits.append((s_d_d / self.temperature).masked_fill(~mask_dd, neg_inf).reshape(B, -1))

        if negative_passages is not None:
            d_neg = F.normalize(negative_passages, dim=-1)
            s_hard = torch.einsum("bd,bkd->bk", q, d_neg)
            mask_hard = (s_hard <= threshold)
            logits.append((s_hard / self.temperature).masked_fill(~mask_hard, neg_inf))

        logits_all = torch.cat(logits, dim=1)
        logZ = torch.logsumexp(logits_all, dim=1)
        loss = - (s_pos / self.temperature) + logZ
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

        self.test_query_embs = []
        self.test_passage_embs = []

    def forward(self, x):
        query = self.tokenizer(x["question"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        passage = self.tokenizer(x["context"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_output = self.model(**query)
        passage_output = self.model(**passage)
        query_emb = pool(query_output.last_hidden_state, query["attention_mask"], pooling_method="mean")
        passage_emb = pool(passage_output.last_hidden_state, passage["attention_mask"],pooling_method="mean")
        
        if isinstance(self.loss_fn, HardNegativeInfoNCELoss):
            hard_negatives = []
            for i in range(len(x["hard_negatives"])):
                tokens = self.tokenizer(x["hard_negatives"][i], return_tensors="pt", truncation=True, padding=True).to(self.device)
                output = self.model(**tokens)
                output_emb = pool(output.last_hidden_state, tokens["attention_mask"], pooling_method="mean")
                hard_negatives.append(output_emb)

            hard_negatives_emb = torch.stack(hard_negatives)

            return query_emb, passage_emb, hard_negatives_emb

        elif isinstance(self.loss_fn, ContrastiveLoss):
            return query_emb, passage_emb
    
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
        if isinstance(self.loss_fn, HardNegativeInfoNCELoss):
            query_emb, passage_emb, hard_negatives_emb = self(batch)
            loss = self.loss_fn(query_emb, passage_emb, hard_negatives_emb)
            self.log("train_loss", loss, batch_size=query_emb.shape[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            return loss

        elif isinstance(self.loss_fn, ContrastiveLoss):
            query_emb, passage_emb = self(batch)
            negative_passages = []
            for i in range(len(passage_emb)):
                negatives = torch.cat([passage_emb[:i], passage_emb[i + 1:]])
                negative_passages.append(negatives)
            negative_passages = torch.stack(negative_passages)

            loss = self.loss_fn(query_emb, passage_emb, negative_passages, 0.01)
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

    def on_test_start(self) -> None:
        self.test_query_embs = []
        self.test_passage_embs = []

    def test_step(self, x, batch_idx):
        query = self.tokenizer(x["question"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        passage = self.tokenizer(x["context"], return_tensors="pt", truncation=True, padding=True).to(self.device)
        query_output = self.model(**query)
        passage_output = self.model(**passage)

        query_emb = pool(query_output.last_hidden_state, query["attention_mask"], pooling_method="mean")
        passage_emb = pool(passage_output.last_hidden_state,passage["attention_mask"], pooling_method="mean")

        self.test_query_embs.append(query_emb.detach().cpu())
        self.test_passage_embs.append(passage_emb.detach().cpu())

        return None

    def on_test_epoch_end(self):
        all_query_embs = torch.cat(self.test_query_embs, dim=0).to(self.device)
        all_passage_embs = torch.cat(self.test_passage_embs, dim=0).to(self.device)

        N = all_query_embs.size(0)
        k = 2
        ndcg_scores = []

        chunk_size = 16

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q_chunk = all_query_embs[start:end]

            sim_chunk = F.cosine_similarity(q_chunk.unsqueeze(1), all_passage_embs.unsqueeze(0), dim=-1)

            for local_i in range(sim_chunk.size(0)):
                global_i = start + local_i
                labels = torch.zeros(N, device=self.device)
                labels[global_i] = 1
                ndcg = get_ndcg(sim_chunk[local_i], labels, k=k)
                ndcg_scores.append(torch.tensor(ndcg, device=self.device, dtype=torch.float32))

        ndcg_scores = torch.stack(ndcg_scores)
        self.log("test_ndcg@2_full", ndcg_scores.mean(), prog_bar=True, on_epoch=True)