from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PoolerOutput:
    latents: torch.Tensor
    latent_mask: torch.Tensor
    aux: dict[str, torch.Tensor]


def _pad_to_blocks(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, dim = hidden.shape
    n_blocks = math.ceil(seq_len / block_size)
    padded_len = n_blocks * block_size
    pad_len = padded_len - seq_len
    if pad_len:
        hidden = F.pad(hidden, (0, 0, 0, pad_len), value=0.0)
        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
    block_hidden = hidden.view(bsz, n_blocks, block_size, dim)
    block_mask = attention_mask.view(bsz, n_blocks, block_size).bool()
    return block_hidden, block_mask


def _block_latent_mask(block_mask: torch.Tensor) -> torch.Tensor:
    return block_mask.any(dim=-1).long()


class MeanBlockPooler(nn.Module):
    def __init__(self, input_dim: int, compression_ratio: int):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = input_dim
        self.compression_ratio = compression_ratio

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> PoolerOutput:
        block_hidden, block_mask = _pad_to_blocks(hidden, attention_mask, self.compression_ratio)
        weights = block_mask.to(hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        latents = (block_hidden * weights).sum(dim=2) / denom
        return PoolerOutput(
            latents=latents,
            latent_mask=_block_latent_mask(block_mask),
            aux={},
        )


class AttentionBlockPooler(nn.Module):
    """Learned non-graph attention pooler, useful as a learned-pooling baseline."""

    def __init__(self, input_dim: int, compression_ratio: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = input_dim
        self.compression_ratio = compression_ratio
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> PoolerOutput:
        block_hidden, block_mask = _pad_to_blocks(hidden, attention_mask, self.compression_ratio)
        scores = self.scorer(block_hidden).squeeze(-1)
        scores = scores.masked_fill(~block_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).masked_fill(~block_mask, 0.0)
        latents = torch.sum(weights.unsqueeze(-1) * block_hidden, dim=2)
        return PoolerOutput(
            latents=latents,
            latent_mask=_block_latent_mask(block_mask),
            aux={"pool_weights_mean": weights.mean()},
        )


def _dense_graph_mask(
    block_hidden: torch.Tensor,
    block_mask: torch.Tensor,
    graph: str,
    topk: int,
    tau: float,
    local_edges: bool,
) -> torch.Tensor:
    """Build a dense block-local token graph from hidden-state cosine similarity."""
    bsn, block_size, _ = block_hidden.shape
    device = block_hidden.device
    valid_pair = block_mask.unsqueeze(1) & block_mask.unsqueeze(2)

    h = F.normalize(block_hidden.float(), dim=-1)
    sim = torch.matmul(h, h.transpose(-1, -2))
    sim = sim.masked_fill(~valid_pair, -1e4)

    if graph == "threshold":
        adj = sim > tau
    elif graph == "topk":
        k = max(1, min(topk, block_size))
        top_idx = sim.topk(k=k, dim=-1).indices
        adj = torch.zeros(bsn, block_size, block_size, dtype=torch.bool, device=device)
        adj.scatter_(dim=-1, index=top_idx, value=True)
    elif graph == "complete":
        adj = valid_pair.clone()
    else:
        raise ValueError(f"Unknown graph type: {graph}")

    eye = torch.eye(block_size, dtype=torch.bool, device=device).unsqueeze(0)
    adj = adj | (eye & valid_pair)

    if local_edges and block_size > 1:
        idx = torch.arange(block_size - 1, device=device)
        local = torch.zeros(block_size, block_size, dtype=torch.bool, device=device)
        local[idx, idx + 1] = True
        local[idx + 1, idx] = True
        adj = adj | (local.unsqueeze(0) & valid_pair)

    return adj & valid_pair


class DenseGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.o = nn.Linear(out_dim, out_dim)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        bsn, block_size, _ = x.shape
        q = self.q(x).view(bsn, block_size, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsn, block_size, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsn, block_size, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~adj.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~adj.unsqueeze(1), 0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsn, block_size, self.out_dim)
        out = self.o(out)
        out = self.norm(self.residual(x) + self.dropout(out))
        out = F.gelu(out)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class DenseRepoGraphAttentionLayer(nn.Module):
    """Dense block-local approximation of PyG's GATConv used by GLOT.

    The public GLOT implementation uses torch_geometric.nn.GATConv followed by
    ReLU in the pooler. This layer keeps the same attention form without adding
    torch-geometric as a dependency for 16-token compression blocks.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        bsn, block_size, _ = x.shape
        h = self.lin(x).view(bsn, block_size, self.heads, self.head_dim)
        src_scores = (h * self.att_src.view(1, 1, self.heads, self.head_dim)).sum(dim=-1)
        dst_scores = (h * self.att_dst.view(1, 1, self.heads, self.head_dim)).sum(dim=-1)
        scores = dst_scores.unsqueeze(2) + src_scores.unsqueeze(1)
        scores = F.leaky_relu(scores, negative_slope=0.2).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~adj.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~adj.unsqueeze(1), 0.0)
        attn = self.dropout(attn)
        h_src = h.permute(0, 2, 1, 3)
        out = torch.matmul(attn, h_src).transpose(1, 2).contiguous().view(bsn, block_size, self.out_dim)
        out = out + self.bias
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class DenseGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kind: str = "gcn",
        dropout: float = 0.0,
        repo_style: bool = False,
    ):
        super().__init__()
        self.kind = kind
        self.repo_style = repo_style
        if kind == "gcn":
            self.update = nn.Linear(in_dim, out_dim)
            residual_in = in_dim
        elif kind == "sage":
            self.update = nn.Linear(2 * in_dim, out_dim)
            residual_in = in_dim
        else:
            raise ValueError(f"Unknown dense graph conv kind: {kind}")
        self.residual = nn.Identity() if residual_in == out_dim else nn.Linear(residual_in, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        weights = adj.to(x.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.matmul(weights, x) / denom
        if self.kind == "gcn":
            msg = self.update(neigh)
        else:
            msg = self.update(torch.cat([x, neigh], dim=-1))
        if self.repo_style:
            return self.dropout(msg) * node_mask.unsqueeze(-1).to(msg.dtype)
        out = self.norm(self.residual(x) + self.dropout(msg))
        out = F.gelu(out)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


def _import_pyg_glot() -> dict[str, Any]:
    try:
        from torch_geometric.data import Batch, Data
        from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, MLP
        from torch_geometric.utils import dense_to_sparse, softmax
        from torch_scatter import scatter_add
    except Exception as exc:
        raise ImportError(
            "Exact GLOT requires PyTorch Geometric. Install it on the cluster, for example: "
            "`pip install torch-geometric` and "
            "`pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html`."
        ) from exc
    return {
        "Batch": Batch,
        "Data": Data,
        "GATConv": GATConv,
        "GCNConv": GCNConv,
        "GINConv": GINConv,
        "GINEConv": GINEConv,
        "MLP": MLP,
        "dense_to_sparse": dense_to_sparse,
        "softmax": softmax,
        "scatter_add": scatter_add,
    }


class PyGGLOTBlockPooler(nn.Module):
    """GLOT repo implementation adapted from sentence pooling to LCLM blocks.

    This follows the public GLOT code path: build PyG threshold graphs from
    token hidden states, apply PyG graph convolution layers, ReLU after each
    layer, concatenate Jumping-Knowledge features, and use a learned readout.
    The only LCLM-specific additions are splitting into compression blocks,
    projecting the GLOT output to the adapter input dimension, and optional
    mean-pooling initialization for a fair baseline start.
    """

    def __init__(
        self,
        input_dim: int,
        compression_ratio: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        jk_mode: str = "cat",
        conv: str = "gat",
        adjacency: str = "threshold",
        tau: float = 0.3,
        output_dim: int | None = None,
        init_as_mean: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        pyg = _import_pyg_glot()
        self.Batch = pyg["Batch"]
        self.Data = pyg["Data"]
        self.GATConv = pyg["GATConv"]
        self.GCNConv = pyg["GCNConv"]
        self.GINConv = pyg["GINConv"]
        self.GINEConv = pyg["GINEConv"]
        self.MLP = pyg["MLP"]
        self.dense_to_sparse = pyg["dense_to_sparse"]
        self.pyg_softmax = pyg["softmax"]
        self.scatter_add = pyg["scatter_add"]

        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.conv = conv
        self.adjacency = adjacency
        self.tau = tau
        self.init_as_mean = init_as_mean
        self.device_override = device
        if self.adjacency != "threshold":
            raise ValueError("Exact GLOT implementation currently supports graph='threshold'")

        self.convs = nn.ModuleList()
        last_dim = input_dim
        for _ in range(num_layers):
            if conv == "gat":
                layer = self.GATConv(last_dim, hidden_dim, edge_dim=1)
            elif conv == "gcn":
                layer = self.GCNConv(last_dim, hidden_dim)
            elif conv == "gine":
                mlp = self.MLP([last_dim, hidden_dim, hidden_dim])
                layer = self.GINEConv(nn=mlp, train_eps=False, edge_dim=1)
            elif conv == "gin":
                mlp = self.MLP([last_dim, hidden_dim, hidden_dim])
                layer = self.GINConv(nn=mlp, train_eps=False)
            else:
                raise ValueError(f"Unknown GLOT conv: {conv}")
            self.convs.append(layer)
            last_dim = hidden_dim

        if jk_mode == "cat":
            self.glot_out_dim = input_dim + num_layers * hidden_dim
        elif jk_mode in {"max", "mean"}:
            self.glot_out_dim = hidden_dim
        else:
            raise ValueError("Unknown JK mode")

        self.score_layer = nn.Sequential(
            nn.Linear(self.glot_out_dim, max(128, self.glot_out_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(128, self.glot_out_dim // 2), 1),
        )
        self.out_dim = int(output_dim) if output_dim is not None else self.glot_out_dim
        self.output_proj = (
            nn.Identity()
            if self.out_dim == self.glot_out_dim
            else nn.Linear(self.glot_out_dim, self.out_dim)
        )
        if self.init_as_mean:
            self._init_as_mean_pooler()

    def _init_as_mean_pooler(self) -> None:
        if self.jk_mode != "cat" or self.out_dim != self.input_dim:
            raise ValueError("init_as_mean requires jk='cat' and output_dim equal to input_dim")
        if not isinstance(self.output_proj, nn.Linear):
            raise ValueError("init_as_mean requires a linear output projection")
        with torch.no_grad():
            nn.init.zeros_(self.score_layer[-1].weight)
            nn.init.zeros_(self.score_layer[-1].bias)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            eye = torch.eye(
                self.input_dim,
                dtype=self.output_proj.weight.dtype,
                device=self.output_proj.weight.device,
            )
            self.output_proj.weight[:, : self.input_dim].copy_(eye)

    def _threshold_edges(self, sim: torch.Tensor):
        adj = (sim > self.tau).float()
        return self.dense_to_sparse(adj)

    def _build_pyg_graphs(self, hidden: torch.Tensor, attention_mask: torch.Tensor):
        assert hidden.dim() == 3 and attention_mask.dim() == 2, "Bad input shapes"
        _, length, _ = hidden.shape
        device = self.device_override or hidden.device
        graphs = []
        for batch_idx in range(hidden.size(0)):
            mask_b = attention_mask[batch_idx].to(dtype=torch.bool)
            x_b = hidden[batch_idx, mask_b]
            token_idx = torch.arange(length, device=device)[mask_b]
            if x_b.size(0) == 0:
                x_b = hidden.new_zeros(1, hidden.size(-1))
                token_idx = torch.zeros(1, dtype=torch.long, device=device)
            sim = F.cosine_similarity(x_b.unsqueeze(1), x_b.unsqueeze(0), dim=-1)
            edge_index, edge_weight = self._threshold_edges(sim)
            data = self.Data(x=x_b, edge_index=edge_index, edge_attr=edge_weight).to(device)
            data.token_idx = token_idx
            graphs.append(data)
        return self.Batch.from_data_list(graphs)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> PoolerOutput:
        block_hidden, block_mask = _pad_to_blocks(hidden, attention_mask, self.compression_ratio)
        bsz, n_blocks, block_size, dim = block_hidden.shape
        flat_hidden = block_hidden.reshape(bsz * n_blocks, block_size, dim)
        flat_mask = block_mask.reshape(bsz * n_blocks, block_size)

        batch = self._build_pyg_graphs(flat_hidden, flat_mask).to(flat_hidden.device)
        x, edge_index = batch.x, batch.edge_index
        edge_weight = getattr(batch, "edge_attr", None)

        h_list = [x]
        h = x
        for conv in self.convs:
            if isinstance(conv, self.GATConv):
                h = conv(h, edge_index, edge_attr=edge_weight)
            elif isinstance(conv, self.GCNConv):
                h = conv(h, edge_index, edge_weight=edge_weight.squeeze())
            elif isinstance(conv, self.GINConv):
                h = conv(h, edge_index)
            elif isinstance(conv, self.GINEConv):
                h = conv(h, edge_index, edge_attr=edge_weight)
            h = F.relu(h)
            h_list.append(h)

        if self.jk_mode == "cat":
            h_all = torch.cat(h_list, dim=-1)
        elif self.jk_mode == "max":
            h_all = torch.stack(h_list[1:], dim=-1).max(dim=-1).values
        elif self.jk_mode == "mean":
            h_all = torch.stack(h_list[1:], dim=-1).mean(dim=-1)
        else:
            raise ValueError("Unknown JK mode")

        scores = self.score_layer(h_all).squeeze(-1)
        weights = self.pyg_softmax(scores, batch.batch)
        pooled = self.scatter_add(weights.unsqueeze(-1) * h_all, batch.batch, dim=0)
        pooled = self.output_proj(pooled)
        latents = pooled.view(bsz, n_blocks, self.out_dim)

        valid_counts = flat_mask.sum(dim=-1).float().clamp_min(1.0)
        edge_density = torch.tensor(
            float(edge_index.size(1)),
            device=flat_hidden.device,
            dtype=flat_hidden.dtype,
        ) / (valid_counts.square().sum().to(flat_hidden.dtype).clamp_min(1.0))
        entropy_terms = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log())
        entropy = self.scatter_add(
            entropy_terms,
            batch.batch,
            dim=0,
            dim_size=bsz * n_blocks,
        ).mean()

        return PoolerOutput(
            latents=latents,
            latent_mask=_block_latent_mask(block_mask),
            aux={"edge_density": edge_density, "pool_entropy": entropy},
        )


class GLOTBlockPooler(nn.Module):
    """GLOT-style block-local token graph pooling.

    For each compression block, construct a cosine-similarity graph over token hidden states,
    refine node features with dense GAT layers, then use a learned readout over nodes to
    produce one latent token.
    """

    def __init__(
        self,
        input_dim: int,
        compression_ratio: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        graph: str = "topk",
        topk: int = 4,
        tau: float = 0.25,
        local_edges: bool = True,
        dropout: float = 0.0,
        jk: str = "cat",
        gnn_type: str = "gat",
        output_dim: int | None = None,
        residual_mean: bool = False,
        zero_init_output: bool = False,
        init_as_mean: bool = False,
        layer_style: str = "stable",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.graph = graph
        self.topk = topk
        self.tau = tau
        self.local_edges = local_edges
        self.jk = jk
        self.gnn_type = gnn_type
        self.residual_mean = residual_mean
        self.init_as_mean = init_as_mean
        self.layer_style = layer_style
        if self.layer_style not in {"stable", "repo"}:
            raise ValueError("layer_style must be 'stable' or 'repo'")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            if gnn_type == "gat":
                if self.layer_style == "repo":
                    layers.append(DenseRepoGraphAttentionLayer(in_dim, hidden_dim, heads=heads, dropout=dropout))
                else:
                    layers.append(DenseGraphAttentionLayer(in_dim, hidden_dim, heads=heads, dropout=dropout))
            elif gnn_type in {"gcn", "sage"}:
                layers.append(
                    DenseGraphConvLayer(
                        in_dim,
                        hidden_dim,
                        kind=gnn_type,
                        dropout=dropout,
                        repo_style=self.layer_style == "repo",
                    )
                )
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        if jk == "cat":
            readout_dim = input_dim + num_layers * hidden_dim
        elif jk == "last":
            readout_dim = hidden_dim
        else:
            raise ValueError(f"Unknown jk mode: {jk}")
        self.readout_dim = readout_dim
        self.out_dim = int(output_dim) if output_dim is not None else readout_dim
        if self.residual_mean and self.out_dim != self.input_dim:
            raise ValueError("residual_mean requires output_dim to equal input_dim")
        if self.init_as_mean and (self.jk != "cat" or self.out_dim != self.input_dim):
            raise ValueError("init_as_mean requires jk='cat' and output_dim equal to input_dim")

        scorer_hidden = max(128, min(1024, self.readout_dim // 2))
        self.scorer = nn.Sequential(
            nn.Linear(self.readout_dim, scorer_hidden),
            nn.Tanh(),
            nn.Linear(scorer_hidden, 1),
        )
        if self.out_dim == self.readout_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(self.readout_dim, self.out_dim)
            if zero_init_output:
                nn.init.zeros_(self.output_proj.weight)
                nn.init.zeros_(self.output_proj.bias)
        if self.init_as_mean:
            self._init_as_mean_pooler()

    def _init_as_mean_pooler(self) -> None:
        if not isinstance(self.output_proj, nn.Linear):
            raise ValueError("init_as_mean requires a linear output projection")
        with torch.no_grad():
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            eye = torch.eye(
                self.input_dim,
                dtype=self.output_proj.weight.dtype,
                device=self.output_proj.weight.device,
            )
            self.output_proj.weight[:, : self.input_dim].copy_(eye)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> PoolerOutput:
        block_hidden, block_mask = _pad_to_blocks(hidden, attention_mask, self.compression_ratio)
        bsz, n_blocks, block_size, dim = block_hidden.shape
        flat_hidden = block_hidden.reshape(bsz * n_blocks, block_size, dim)
        flat_mask = block_mask.reshape(bsz * n_blocks, block_size)

        adj = _dense_graph_mask(
            flat_hidden,
            flat_mask,
            graph=self.graph,
            topk=self.topk,
            tau=self.tau,
            local_edges=self.local_edges,
        )

        h = flat_hidden
        h_list = [h]
        for layer in self.layers:
            h = layer(h, adj=adj, node_mask=flat_mask)
            if self.layer_style == "repo":
                h = F.relu(h)
            h_list.append(h)

        if self.jk == "cat":
            readout_hidden = torch.cat(h_list, dim=-1)
        else:
            readout_hidden = h_list[-1]

        scores = self.scorer(readout_hidden).squeeze(-1)
        scores = scores.masked_fill(~flat_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).masked_fill(~flat_mask, 0.0)
        pooled = torch.sum(weights.unsqueeze(-1) * readout_hidden, dim=1)
        pooled = self.output_proj(pooled)
        if self.residual_mean:
            mean_weights = flat_mask.to(flat_hidden.dtype).unsqueeze(-1)
            denom = mean_weights.sum(dim=1).clamp_min(1.0)
            mean_pooled = (flat_hidden * mean_weights).sum(dim=1) / denom
            pooled = pooled + mean_pooled
        latents = pooled.view(bsz, n_blocks, self.out_dim)

        edge_density = adj.float().mean()
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return PoolerOutput(
            latents=latents,
            latent_mask=_block_latent_mask(block_mask),
            aux={"edge_density": edge_density, "pool_entropy": entropy},
        )


def build_pooler(input_dim: int, cfg: dict) -> nn.Module:
    ratio = int(cfg["ratio"])
    name = cfg["pooler"]
    if name == "mean":
        return MeanBlockPooler(input_dim=input_dim, compression_ratio=ratio)
    if name == "attention":
        return AttentionBlockPooler(input_dim=input_dim, compression_ratio=ratio)
    if name == "glot":
        glot_cfg = cfg.get("glot", {})
        if str(glot_cfg.get("implementation", "dense")) == "pyg":
            return PyGGLOTBlockPooler(
                input_dim=input_dim,
                compression_ratio=ratio,
                hidden_dim=int(glot_cfg.get("hidden_dim", 128)),
                num_layers=int(glot_cfg.get("num_layers", 2)),
                jk_mode=str(glot_cfg.get("jk", "cat")),
                conv=str(glot_cfg.get("conv", glot_cfg.get("gnn_type", "gat"))),
                adjacency=str(glot_cfg.get("graph", "threshold")),
                tau=float(glot_cfg.get("tau", 0.3)),
                output_dim=glot_cfg.get("output_dim"),
                init_as_mean=bool(glot_cfg.get("init_as_mean", False)),
            )
        return GLOTBlockPooler(
            input_dim=input_dim,
            compression_ratio=ratio,
            hidden_dim=int(glot_cfg.get("hidden_dim", 256)),
            num_layers=int(glot_cfg.get("num_layers", 2)),
            heads=int(glot_cfg.get("heads", 4)),
            graph=str(glot_cfg.get("graph", "topk")),
            topk=int(glot_cfg.get("topk", 4)),
            tau=float(glot_cfg.get("tau", 0.25)),
            local_edges=bool(glot_cfg.get("local_edges", True)),
            dropout=float(glot_cfg.get("dropout", 0.0)),
            jk=str(glot_cfg.get("jk", "cat")),
            gnn_type=str(glot_cfg.get("gnn_type", "gat")),
            output_dim=glot_cfg.get("output_dim"),
            residual_mean=bool(glot_cfg.get("residual_mean", False)),
            zero_init_output=bool(glot_cfg.get("zero_init_output", False)),
            init_as_mean=bool(glot_cfg.get("init_as_mean", False)),
            layer_style=str(glot_cfg.get("layer_style", "stable")),
        )
    raise ValueError(f"Unknown pooler: {name}")
