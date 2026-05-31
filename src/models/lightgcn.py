"""
DataFlix — Model 4: LightGCN (Graph Collaborative Filtering)
src/models/lightgcn.py
"""
import logging, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

from src.models.hybrid import TripletDataset
from src.config import DEVICE, RESULTS_DIR, LATENT_DIM_K, BPR_BATCH_SIZE, BPR_SAMPLES_PER_EPOCH, LR_BPR

log = logging.getLogger(__name__)
LIGHTGCN_CKPT_PATH = RESULTS_DIR / "lightgcn_best.pt"

class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int = LATENT_DIM_K, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers, self.embed_dim = n_users, n_items, n_layers, embed_dim
        self.embedding = nn.Embedding(n_users + n_items, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor):
        embs = [self.embedding.weight]
        emb = self.embedding.weight
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(edge_index, emb)
            embs.append(emb)
        final_embs = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(final_embs, [self.n_users, self.n_items])

class LightGCNTrainer:
    def __init__(
        self, model: LightGCN, adj_matrix_path: Path, all_items: np.ndarray, item_pop: np.ndarray,
        device=DEVICE, lr=LR_BPR, reg=1e-4, n_epochs=50, batch_size=BPR_BATCH_SIZE, samples=BPR_SAMPLES_PER_EPOCH
    ):
        # ---> PROFESSIONAL MLOPS: CPU OFFLOADING <---
        self.cpu_device = torch.device('cpu')
        self.gpu_device = device

        # Keep the massive graph and model in System RAM
        self.model = model.to(self.cpu_device)
        self.edge_index = self._load_graph(adj_matrix_path).to(self.cpu_device)

        self.n_epochs, self.batch_size, self.samples = n_epochs, batch_size, samples
        self.all_items, self.item_pop, self.reg = all_items, item_pop, reg
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.best_loss = float("inf")

    def _load_graph(self, path: Path):
        log.info("Loading precomputed LightGCN graph into System RAM...")
        norm_adj = sp.load_npz(path).tocoo()
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))

    def fit(self, train_pos_sets):
        log.info(f"Training LightGCN: {self.n_epochs} epochs | Strategy: CPU-Graph -> GPU-Loss")
        eligible_users = np.array(list(train_pos_sets.keys()), dtype=np.int32)

        for epoch in range(1, self.n_epochs + 1):
            t, total_loss, n_batches = time.time(), 0.0, 0
            self.model.train()
            loader = DataLoader(TripletDataset(train_pos_sets, self.all_items, self.item_pop, self.samples, eligible_users), batch_size=self.batch_size, shuffle=True, num_workers=0)

            for u, pos, neg in loader:
                # 1. Forward pass strictly on CPU System RAM
                user_embs, item_embs = self.model(self.edge_index)
                u_emb, pos_emb, neg_emb = user_embs[u], item_embs[pos], item_embs[neg]

                # 2. Transfer ONLY the batch slices to GPU for loss calc
                u_emb_g = u_emb.to(self.gpu_device)
                pos_emb_g = pos_emb.to(self.gpu_device)
                neg_emb_g = neg_emb.to(self.gpu_device)

                bpr_loss = -torch.nn.functional.logsigmoid((u_emb_g * pos_emb_g).sum(1) - (u_emb_g * neg_emb_g).sum(1)).mean()

                # Reg loss (Transfer base embeddings to GPU)
                u_base = self.model.embedding(u).to(self.gpu_device)
                pos_base = self.model.embedding(pos + self.model.n_users).to(self.gpu_device)
                neg_base = self.model.embedding(neg + self.model.n_users).to(self.gpu_device)
                reg_loss = (u_base.norm(2).pow(2) + pos_base.norm(2).pow(2) + neg_base.norm(2).pow(2)) / float(len(u))

                loss = bpr_loss + self.reg * reg_loss

                # 3. Backward Pass (Seamlessly travels from GPU back to CPU)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            log.info(f"  Epoch {epoch:>3}/{self.n_epochs} | loss={total_loss/n_batches:.5f} | ({time.time()-t:.1f}s)")

        self.save()
        return self

    def save(self, path=LIGHTGCN_CKPT_PATH):
        torch.save({"model_state": self.model.state_dict(), "n_users": self.model.n_users, "n_items": self.model.n_items, "embed_dim": self.model.embed_dim, "n_layers": self.model.n_layers}, path)

@torch.no_grad()
def score_lightgcn(model: LightGCN, user_idx: int, edge_index: torch.Tensor, device=DEVICE) -> np.ndarray:
    # Evaluate on CPU to prevent evaluation VRAM crashes
    model = model.cpu()
    edge_index = edge_index.cpu()
    
    user_embs, item_embs = model(edge_index)
    u_vec = user_embs[user_idx].unsqueeze(0)
    scores = (u_vec * item_embs).sum(dim=1)
    return scores.numpy()