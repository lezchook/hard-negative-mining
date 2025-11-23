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
    def __init__(self, temperature: float = 0.02, false_negative_margin: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.false_negative_margin = false_negative_margin
        self.eps = eps

    def forward(self, query: torch.Tensor, passage: torch.Tensor, negative_passages: torch.Tensor = None):
        device = query.device
        batch_size = query.size(0)

        q = F.normalize(query, dim=-1)
        d_pos = F.normalize(passage, dim=-1)

        if negative_passages is not None:
            d_neg = F.normalize(negative_passages, dim=-1)

        s_pos = (q * d_pos).sum(dim=-1)

        if negative_passages is not None:
            s_hard = torch.einsum("bd,bkd->bk", q, d_neg)
        else:
            s_hard = None

        s_q_q = q @ q.t()
        s_q_d = q @ d_pos.t()
        s_dpos_d = d_pos @ d_pos.t()

        threshold = s_pos.unsqueeze(1) + self.false_negative_margin

        if s_hard is not None:
            mask_hard = (s_hard <= threshold).float()
        else:
            mask_hard = None


        eye = torch.eye(batch_size, device=device)
        mask_qd = (s_q_d <= threshold).float()
        mask_qd = mask_qd * (1.0 - eye)

        mask_qq = (s_q_q <= threshold).float()
        mask_qq = mask_qq * (1.0 - eye)

        s_pos_scaled = s_pos / self.temperature
        if s_hard is not None:
            s_hard_scaled = s_hard / self.temperature
        s_q_q_scaled = s_q_q / self.temperature
        s_q_d_scaled = s_q_d / self.temperature
        s_dpos_d_scaled = s_dpos_d / self.temperature

        exp_pos = torch.exp(s_pos_scaled)

        if s_hard is not None:
            exp_hard = torch.exp(s_hard_scaled) * mask_hard
            sum_hard = exp_hard.sum(dim=1)
        else:
            sum_hard = torch.zeros_like(exp_pos)

        exp_q_q = torch.exp(s_q_q_scaled) * mask_qq
        sum_q_q = exp_q_q.sum(dim=1)

        exp_d_d = torch.exp(s_dpos_d_scaled) * mask_qd
        sum_d_d = exp_d_d.sum(dim=1)

        exp_q_d = torch.exp(s_q_d_scaled) * mask_qd
        sum_q_d = exp_q_d.sum(dim=1)

        Z = exp_pos + sum_hard + sum_q_q + sum_d_d + sum_q_d

        loss = -torch.log(exp_pos / (Z + self.eps))
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