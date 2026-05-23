# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class BaseSheafConv(MessagePassing):
    """Shared base for all NSD and NSP sheaf convolution layers.

    Factors out the parameterization and utilities that are identical across
    every variant in both model families:

    * ``W1`` / ``W2`` -- bilateral stalk transforms (left d*d, right f*f).
    * ``sigma`` -- activation function (Tanh).
    * ``reset_parameters()`` -- Xavier init for W1, W2, and any map_generator.
    * ``_apply_stalk_transform(x)`` -- computes ``W1 @ x @ W2``.
    * ``_compute_s_norm(edge_index, num_nodes, dtype)`` -- symmetric degree
      normalisation coefficients ``1 / sqrt(deg(v) * deg(u))`` per edge.

    Concrete subclasses must implement:
        ``get_map_products(x_feat, edge_index) -> (self_map, cross_map)``
        ``forward(x_feat, x_stalk, edge_index) -> updated stalk``
        ``message(...)``
    """

    def __init__(
        self,
        stalk_dim: int,
        in_channels: int,
        hidden_dim: int,
        context_dim: int | None = None,
        add_self_loops: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.stalk_dim = stalk_dim
        self.in_channels = in_channels  # 'f' (feature dimension per stalk entry)
        self.context_dim = (
            context_dim if context_dim is not None else (stalk_dim * in_channels)
        )
        self.add_self_loops = add_self_loops

        self.W1 = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.W2 = nn.Parameter(torch.empty(in_channels, in_channels))  # [f, f]
        self.sigma = nn.Tanh()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        # Handles both the singular map_generator and a map_generators ModuleList.
        generators: list[nn.Sequential] = []
        if hasattr(self, "map_generator"):
            gen = self.map_generator
            if isinstance(gen, nn.Sequential):
                generators.append(gen)
        if hasattr(self, "map_generators"):
            map_gens = self.map_generators
            if isinstance(map_gens, nn.ModuleList):
                for gen in map_gens:
                    if isinstance(gen, nn.Sequential):
                        generators.append(gen)
        for gen in generators:
            for m in gen:
                if isinstance(m, nn.Linear):
                    # gain=0.01: warm-start near-zero so the Laplacian is off at init.
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def _apply_stalk_transform(self, x):
        """Applies bilateral stalk transform: W1 @ x @ W2."""
        return torch.matmul(torch.matmul(self.W1, x), self.W2)

    def _compute_s_norm(self, edge_index, num_nodes, dtype):
        """Computes symmetric degree normalisation coefficients per edge.

        When ``add_self_loops`` is ``True``, self-loops are added before
        computing the degree so that every node has degree at least 1, which
        avoids division by zero for isolated nodes. The normalization uses the
        augmented degree but is returned only for the original edges.
        """
        src_idx, dst_idx = edge_index
        if self.add_self_loops:
            edge_index_for_deg, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        else:
            edge_index_for_deg = edge_index
        deg = degree(edge_index_for_deg[1], num_nodes, dtype=dtype)
        norm = deg.pow(-0.5)
        norm[deg == 0] = 0.0
        return norm[dst_idx] * norm[src_idx]

    def message(  # ty: ignore[invalid-method-override]
        self, z_dst, z_src, self_map, cross_map, s_norm
    ):
        """Builds per-edge sheaf Laplacian messages.

        Args:
            z_dst: Destination-node transformed stalks [E, d, f].
            z_src: Source-node transformed stalks [E, d, f].
            self_map: Precomputed restriction_maps_dst^T restriction_maps_dst
                per edge [E, d, d].
            cross_map: Precomputed restriction_maps_dst^T restriction_maps_src
                per edge [E, d, d].
            s_norm: Symmetric normalization coefficient [E].

        Returns:
            torch.Tensor: Per-edge messages [E, d, f].
        """
        return s_norm.view(-1, 1, 1) * (
            torch.matmul(self_map, z_dst) - torch.matmul(cross_map, z_src)
        )


__all__ = ["BaseSheafConv"]
