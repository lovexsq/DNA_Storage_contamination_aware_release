from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        aggregated = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), device=x.device)
        aggregated.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones((src.shape[0], 1), device=x.device))
        neighbor_mean = aggregated / degree.clamp(min=1.0)
        out = self.self_linear(x) + self.neighbor_linear(neighbor_mean)
        return self.norm(F.relu(out) + x)


class GCNLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        aggregated = torch.zeros_like(x)
        degree = torch.ones((x.shape[0], 1), device=x.device)
        aggregated += x
        aggregated.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones((src.shape[0], 1), device=x.device))
        out = self.linear(aggregated / degree.clamp(min=1.0))
        return self.norm(F.relu(out) + x)


class MultiHeadGATLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim 必须能被 heads 整除。")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.edge_linear = nn.Linear(edge_dim, hidden_dim)
        self.attn = nn.Linear(self.head_dim * 3, 1)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _destination_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        index = dst[:, None].expand(-1, scores.shape[1])
        max_per_node = torch.full(
            (num_nodes, scores.shape[1]),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        max_per_node.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        shifted = scores - max_per_node.gather(0, index)
        exp_scores = torch.exp(shifted)
        denom = torch.zeros_like(max_per_node)
        denom.scatter_add_(0, index, exp_scores)
        return exp_scores / denom.gather(0, index).clamp_min(1e-9)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        projected = self.node_linear(x).view(x.shape[0], self.heads, self.head_dim)
        edge_context = self.edge_linear(edge_features).view(edge_features.shape[0], self.heads, self.head_dim)
        attention_input = torch.cat([projected[src], projected[dst], edge_context], dim=2)
        scores = self.attn(attention_input).squeeze(2)

        if hasattr(scores, "scatter_reduce_"):
            weights = self._destination_softmax(scores, dst, x.shape[0])
        else:
            weights = torch.zeros_like(scores)
            for node_id in torch.unique(dst):
                mask = dst == node_id
                weights[mask] = torch.softmax(scores[mask], dim=0)

        messages = projected[src] * weights.unsqueeze(2)
        aggregated = torch.zeros_like(x)
        aggregated_heads = aggregated.view(x.shape[0], self.heads, self.head_dim)
        aggregated_heads.index_add_(0, dst, messages)
        out = F.relu(self.output_linear(aggregated_heads.reshape(x.shape[0], -1)))
        return self.norm(out + x)


class EdgeAwareMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        raw_messages = self.message(torch.cat([x[src], edge_features], dim=1))
        gates = self.gate(torch.cat([x[src], x[dst], edge_features], dim=1))
        messages = raw_messages * gates

        aggregated = torch.zeros_like(x)
        degree = torch.zeros((x.shape[0], 1), device=x.device)
        aggregated.index_add_(0, dst, messages)
        degree.index_add_(0, dst, torch.ones((src.shape[0], 1), device=x.device))
        aggregated = aggregated / degree.clamp(min=1.0)

        out = self.update(torch.cat([x, aggregated], dim=1))
        return self.norm(out + x)


class EdgePredictor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 96,
        layers: int = 4,
        backbone: Literal["sage", "gcn", "gat", "mpnn"] = "gat",
        dropout: float = 0.15,
        heads: int = 4,
        use_read_head: bool = False,
        read_feature_dim: int = 5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_read_head = use_read_head
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        gnn_layers = []
        for _ in range(layers):
            if backbone == "sage":
                gnn_layers.append(GraphSAGELayer(hidden_dim))
            elif backbone == "gcn":
                gnn_layers.append(GCNLayer(hidden_dim))
            elif backbone == "gat":
                gnn_layers.append(MultiHeadGATLayer(hidden_dim, edge_dim, heads=heads))
            elif backbone == "mpnn":
                gnn_layers.append(EdgeAwareMPNNLayer(hidden_dim, edge_dim, dropout=dropout))
            else:
                raise ValueError(f"未知 backbone: {backbone}")
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.read_head = (
            nn.Sequential(
                nn.Linear(read_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            if use_read_head
            else None
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.encoder(node_features)
        for layer in self.gnn_layers:
            if isinstance(layer, (MultiHeadGATLayer, EdgeAwareMPNNLayer)):
                x = layer(x, edge_index, edge_features)
            else:
                x = layer(x, edge_index)
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], torch.abs(x[src] - x[dst]), edge_features], dim=1)
        return self.edge_head(edge_input).squeeze(1)

    def predict_read_contamination(self, read_features: torch.Tensor) -> torch.Tensor:
        if self.read_head is None:
            raise RuntimeError("当前模型没有启用 read contamination head。")
        return self.read_head(read_features).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(labels > 0.5, self.alpha, 1.0 - self.alpha)
        return (-alpha_t * (1.0 - pt).pow(self.gamma) * torch.log(pt.clamp(min=1e-8))).mean()
