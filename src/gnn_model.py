"""
Graph Neural Network Architectures for Music Structure Graphs (Task 2)

Implements:
- GraphSAGE
- Graph Attention Network (GAT)
- Graph-level mean pooling
- Multi-label music-tag classification

Input node features:
    32 dimensions
    = 12 chroma + 20 MFCC

Output:
    128-dimensional graph representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer with weighted mean aggregation.

    h_i' = ReLU(
        W [h_i || mean(h_j)]
    )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(
            2 * in_features,
            out_features,
            bias=bias,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:

        num_nodes = x.size(0)

        if edge_index.numel() == 0:
            neighbor_mean = torch.zeros_like(x)
        else:
            src = edge_index[0]
            dst = edge_index[1]

            # -------------------------------------------------
            # Edge weights
            # -------------------------------------------------
            if edge_weight is None:
                weights = torch.ones(
                    src.size(0),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                weights = edge_weight.to(
                    device=x.device,
                    dtype=x.dtype,
                )

            weights = torch.clamp(
                weights,
                min=0.0,
            )

            # -------------------------------------------------
            # Weighted neighbor aggregation
            # -------------------------------------------------
            messages = (
                x[src]
                * weights.unsqueeze(-1)
            )

            aggregated = torch.zeros(
                num_nodes,
                x.size(1),
                dtype=x.dtype,
                device=x.device,
            )

            aggregated.index_add_(
                0,
                dst,
                messages,
            )

            degree = torch.zeros(
                num_nodes,
                dtype=x.dtype,
                device=x.device,
            )

            degree.index_add_(
                0,
                dst,
                weights,
            )

            degree = torch.clamp(
                degree,
                min=1e-6,
            )

            neighbor_mean = (
                aggregated
                / degree.unsqueeze(-1)
            )

        # -----------------------------------------------------
        # Self representation + neighbor representation
        # -----------------------------------------------------
        combined = torch.cat(
            [x, neighbor_mean],
            dim=-1,
        )

        return self.activation(
            self.linear(combined)
        )


class GATLayer(nn.Module):
    """
    Multi-head Graph Attention Network layer.

    Attention is normalized over incoming edges
    for each destination node.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if out_features % heads != 0:
            raise ValueError(
                f"out_features ({out_features}) must be "
                f"divisible by heads ({heads})."
            )

        self.heads = heads
        self.head_dim = out_features // heads
        self.out_features = out_features

        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=False,
        )

        self.att_src = nn.Parameter(
            torch.empty(
                1,
                heads,
                self.head_dim,
            )
        )

        self.att_dst = nn.Parameter(
            torch.empty(
                1,
                heads,
                self.head_dim,
            )
        )

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.linear.weight
        )

        nn.init.xavier_uniform_(
            self.att_src
        )

        nn.init.xavier_uniform_(
            self.att_dst
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:

        num_nodes = x.size(0)

        # -----------------------------------------------------
        # Handle graph with no edges
        # -----------------------------------------------------
        if edge_index.numel() == 0:
            h = self.linear(x)

            return F.elu(h)

        src = edge_index[0]
        dst = edge_index[1]

        # -----------------------------------------------------
        # Project node features
        # -----------------------------------------------------
        h = self.linear(x)

        h = h.view(
            num_nodes,
            self.heads,
            self.head_dim,
        )

        # -----------------------------------------------------
        # Attention coefficients
        # -----------------------------------------------------
        alpha_src = (
            h * self.att_src
        ).sum(dim=-1)

        alpha_dst = (
            h * self.att_dst
        ).sum(dim=-1)

        attention_logits = (
            alpha_src[src]
            + alpha_dst[dst]
        )

        attention_logits = self.leaky_relu(
            attention_logits
        )

        # -----------------------------------------------------
        # Include acoustic edge weights if available
        # -----------------------------------------------------
        if edge_weight is not None:

            weights = edge_weight.to(
                device=x.device,
                dtype=x.dtype,
            )

            weights = torch.clamp(
                weights,
                min=1e-6,
            )

            attention_logits = (
                attention_logits
                + torch.log(weights).unsqueeze(-1)
            )

        # -----------------------------------------------------
        # Stable softmax over incoming edges
        # -----------------------------------------------------
        max_per_node = torch.full(
            (num_nodes, self.heads),
            -float("inf"),
            dtype=x.dtype,
            device=x.device,
        )

        max_per_node.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand(
                -1,
                self.heads,
            ),
            attention_logits,
            reduce="amax",
            include_self=True,
        )

        exp_attention = torch.exp(
            attention_logits
            - max_per_node[dst]
        )

        denominator = torch.zeros(
            num_nodes,
            self.heads,
            dtype=x.dtype,
            device=x.device,
        )

        denominator.index_add_(
            0,
            dst,
            exp_attention,
        )

        alpha = (
            exp_attention
            / (denominator[dst] + 1e-8)
        )

        alpha = self.dropout(
            alpha
        )

        # -----------------------------------------------------
        # Message passing
        # -----------------------------------------------------
        messages = (
            h[src]
            * alpha.unsqueeze(-1)
        )

        output = torch.zeros(
            num_nodes,
            self.heads,
            self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )

        output.index_add_(
            0,
            dst,
            messages,
        )

        output = output.reshape(
            num_nodes,
            self.out_features,
        )

        return F.elu(output)


class MusicGNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder for music structure graphs.

    Supported:
        GraphSAGE
        GAT

    Graph-level representation:
        g = mean(h_i)
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        gnn_type: str = "graphsage",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gnn_type = gnn_type.lower()

        if self.gnn_type not in {
            "graphsage",
            "gat",
        }:
            raise ValueError(
                "gnn_type must be 'graphsage' or 'gat'."
            )

        self.dropout = nn.Dropout(
            dropout
        )

        self.layers = nn.ModuleList()

        current_dim = in_dim

        for layer_idx in range(num_layers):

            next_dim = (
                hidden_dim
                if layer_idx < num_layers - 1
                else out_dim
            )

            if self.gnn_type == "gat":

                self.layers.append(
                    GATLayer(
                        current_dim,
                        next_dim,
                        heads=4,
                        dropout=dropout,
                    )
                )

            else:

                self.layers.append(
                    GraphSAGELayer(
                        current_dim,
                        next_dim,
                    )
                )

            current_dim = next_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ):
        """
        Returns:

            h:
                Node embeddings
                (num_nodes, out_dim)

            g:
                Graph embeddings
                (batch_size, out_dim)
        """

        h = x

        # -----------------------------------------------------
        # Message passing
        # -----------------------------------------------------
        for layer_idx, layer in enumerate(
            self.layers
        ):

            if isinstance(
                layer,
                GraphSAGELayer,
            ):
                h = layer(
                    h,
                    edge_index,
                    edge_weight,
                )

            else:
                h = layer(
                    h,
                    edge_index,
                    edge_weight,
                )

            if (
                layer_idx
                < len(self.layers) - 1
            ):
                h = self.dropout(h)

        # -----------------------------------------------------
        # Graph-level mean pooling
        # -----------------------------------------------------
        if batch is None:

            graph_embedding = h.mean(
                dim=0,
                keepdim=True,
            )

        else:

            batch = batch.to(
                device=h.device,
                dtype=torch.long,
            )

            if batch.numel() == 0:
                raise ValueError(
                    "Batch vector cannot be empty."
                )

            batch_size = (
                int(batch.max().item()) + 1
            )

            graph_embedding = torch.zeros(
                batch_size,
                h.size(1),
                dtype=h.dtype,
                device=h.device,
            )

            graph_embedding.index_add_(
                0,
                batch,
                h,
            )

            counts = torch.zeros(
                batch_size,
                dtype=h.dtype,
                device=h.device,
            )

            counts.index_add_(
                0,
                batch,
                torch.ones_like(
                    batch,
                    dtype=h.dtype,
                ),
            )

            graph_embedding = (
                graph_embedding
                / counts.clamp(
                    min=1.0
                ).unsqueeze(-1)
            )

        return h, graph_embedding


class MusicGNNClassifier(nn.Module):
    """
    Task 2 GNN classifier.

    Produces multi-label predictions for the
    50-tag vocabulary.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_tags: int = 50,
        gnn_type: str = "graphsage",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gnn = MusicGNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            out_dim,
            num_tags,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ):

        node_embeddings, graph_embedding = (
            self.gnn(
                x,
                edge_index,
                batch=batch,
                edge_weight=edge_weight,
            )
        )

        dropped = self.dropout(
            graph_embedding
        )

        logits = self.classifier(
            dropped
        )

        probabilities = torch.sigmoid(
            logits
        )

        return {
            "logits": logits,
            "probs": probabilities,
            "g": graph_embedding,
            "node_embeddings": node_embeddings,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
        )