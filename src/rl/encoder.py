"""
Dual-Stream Graph Neural Network Encoder

=== CORE DESIGN PHILOSOPHY ===

The hospital layout optimization problem involves two fundamentally different
types of information that should be processed separately before fusion:

1. Physical Stream:
   - What: Static building geometry and slot properties
   - Index: slot_idx (physical location in building)
   - Data: distance_matrix[slot_i, slot_j], area_vector, position_matrix
   - Property: NEVER changes during optimization

2. Flow Stream:
   - What: Patient movement patterns and department workload
   - Index: dept_idx (functional unit identifier)
   - Data: flow_matrix[dept_i, dept_j], service_times, service_weights
   - Property: Fixed for an episode, changes between episodes

=== KEY INSIGHT ===

The naive approach of concatenating all features into a single GNN fails to
capture the distinct semantics:
- distance_matrix[i, j] is indexed by physical SLOTS
- flow_matrix[i, j] is indexed by functional DEPARTMENTS

When we swap departments, the flow relationships (who needs to visit whom)
stay fixed, but the effective distances change because departments move
to different slots.

This dual-stream architecture:
1. Encodes physical structure with PhysicalStreamEncoder
2. Encodes flow patterns with FlowStreamEncoder
3. Fuses both views with cross-attention to produce layout-aware embeddings

=== MATHEMATICAL FORMULATION ===

Let:
- S = number of slots = number of departments (1:1 mapping)
- D[s_i, s_j] = distance matrix (slot-indexed)
- F[d_i, d_j] = flow matrix (dept-indexed)
- π: dept_idx → slot_idx = current layout mapping

Physical embedding:  h_phys[s] = PhysicalEncoder(slot_features, D)
Flow embedding:      h_flow[d] = FlowEncoder(dept_features, F)

The key step is ALIGNMENT via the layout mapping:
- h_flow_aligned[s] = h_flow[π⁻¹(s)]  # Reorder by slot
- h_fused = CrossAttention(query=h_phys, key=h_flow_aligned, value=h_flow_aligned)

This way, position s gets:
- Physical context from h_phys[s]
- Flow demand from the department currently at slot s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class PhysicalStreamEncoder(nn.Module):
    """
    Encodes static physical/spatial properties of slots using distance-aware attention.

    Uses F.scaled_dot_product_attention with learnable distance bias:
        attn(Q, K, V) = softmax(QK^T/√d + distance_bias) V

    Args:
        slot_feat_dim: Dimension of slot features (default: 4 for area + xyz)
        hidden_dim: Hidden dimension for attention layers
        num_layers: Number of attention layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        num_distance_bins: Number of bins for learnable distance bias
    """

    def __init__(
        self,
        slot_feat_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_distance_bins: int = 16,
    ):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads
        self.num_bins: int = num_distance_bins
        self.dropout: float = dropout

        self.input_projection = nn.Sequential(
            nn.Linear(slot_feat_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        self.distance_bias = nn.Parameter(
            torch.stack(
                [torch.linspace(0, -2, num_distance_bins) for _ in range(num_heads)]
            )
        )

        self.qkv_proj = nn.ModuleList()
        self.out_proj = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.qkv_proj.append(nn.Linear(hidden_dim, hidden_dim * 3))
            self.out_proj.append(nn.Linear(hidden_dim, hidden_dim))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(hidden_dim))
            self.norm2_layers.append(nn.LayerNorm(hidden_dim))

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def _compute_distance_bias(
        self,
        distance_matrix: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert distance matrix to attention bias via learnable binning.

        Returns: (batch, num_heads, n_slots, n_slots)
        """
        batch, n, _ = distance_matrix.shape

        # Normalize to [0, 1]
        max_dist = distance_matrix.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        norm_dist = distance_matrix / max_dist

        # Bin indices: (batch, n, n)
        bin_idx = (norm_dist * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

        # Use embedding lookup: distance_bias.T is (num_bins, num_heads)
        # F.embedding(bin_idx, weight) → (batch, n, n, num_heads)
        attn_bias = F.embedding(
            bin_idx, self.distance_bias.T
        )  # (batch, n, n, num_heads)
        attn_bias = attn_bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)

        # Apply padding mask: invalid positions → -inf
        pad_mask = ~(slot_mask.unsqueeze(1) & slot_mask.unsqueeze(2))  # (batch, n, n)
        attn_bias = attn_bias.masked_fill(pad_mask.unsqueeze(1), float('-inf'))

        return attn_bias

    def forward(
        self,
        slot_features: torch.Tensor,  # (batch, n_slots, slot_feat_dim)
        distance_matrix: torch.Tensor,  # (batch, n_slots, n_slots)
        slot_mask: torch.Tensor,  # (batch, n_slots) - True for valid
    ) -> torch.Tensor:
        """
        Forward pass for physical stream encoding.

        Args:
            slot_features: Normalized slot properties [area, x, y, z]
            distance_matrix: Normalized distance matrix between slots
            slot_mask: Boolean mask for valid slots (True = valid)

        Returns:
            Physical embeddings of shape (batch, n_slots, hidden_dim)
        """
        batch, n, _ = slot_features.shape
        h = self.input_projection(slot_features)

        # Compute distance bias once: (batch, num_heads, n, n)
        attn_bias = self._compute_distance_bias(distance_matrix, slot_mask)

        for i in range(self.num_layers):
            h_normed = self.norm1_layers[i](h)

            # QKV projection and reshape: (batch, n, 3, heads, head_dim)
            qkv = self.qkv_proj[i](h_normed).reshape(
                batch, n, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(
                0
            )  # each: (batch, heads, n, head_dim)

            h_attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, heads, n, head_dim)

            h_attn = h_attn.transpose(1, 2).reshape(batch, n, self.hidden_dim)
            h = h + self.out_proj[i](h_attn)

            h = h + self.ffn_layers[i](self.norm2_layers[i](h))

        return self.output_projection(h)


class FlowStreamEncoder(nn.Module):
    """
    Encodes patient flow patterns between departments.

    This encoder processes:
    1. Department features: [service_time, service_weight]
    2. Flow relationships: how many patients move between departments

    The output captures clinical workflow patterns and department importance,
    which determines what adjacencies are beneficial.

    Architecture:
        Input → Linear → [GCN with flow as edge weights] × L → Output

    Note: We use GCN here because the flow matrix naturally defines
    edge weights (unlike distance which is more like attention bias).

    Args:
        dept_feat_dim: Dimension of department features
        hidden_dim: Hidden dimension for GCN layers
        num_layers: Number of GCN layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        dept_feat_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        self.input_projection = nn.Sequential(
            nn.Linear(dept_feat_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        self.gcn_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _build_normalized_adjacency(
        flow_matrix: torch.Tensor,  # (batch, n, n)
        dept_mask: torch.Tensor,  # (batch, n) - True for valid
    ) -> torch.Tensor:
        """
        Build the symmetric-normalized adjacency with self-loops, per graph.

        Returns:
            A_norm: (batch, n, n) = D^{-1/2}(A+I)D^{-1/2}, zeroed on padding.
        """
        _, n, _ = flow_matrix.shape
        device = flow_matrix.device

        mask_2d = (dept_mask.unsqueeze(1) & dept_mask.unsqueeze(2)).to(flow_matrix.dtype)
        adj = flow_matrix * mask_2d  # drop padding edges

        # Per-graph edge-weight normalization (matches the original max-scaling)
        adj_max = adj.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        adj = adj / adj_max

        # Add self-loops on valid nodes only
        eye = torch.eye(n, device=device, dtype=flow_matrix.dtype).unsqueeze(0)
        adj = adj + eye * dept_mask.unsqueeze(-1).to(flow_matrix.dtype)

        # Symmetric normalization D^{-1/2}(A+I)D^{-1/2}
        deg = adj.sum(dim=-1).clamp(min=1e-6)  # (batch, n)
        d_inv_sqrt = deg.pow(-0.5)  # (batch, n)
        adj_norm = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(1)
        return adj_norm

    def forward(
        self,
        dept_features: torch.Tensor,  # (batch, n_depts, dept_feat_dim)
        flow_matrix: torch.Tensor,  # (batch, n_depts, n_depts)
        dept_mask: torch.Tensor,  # (batch, n_depts) - True for valid
    ) -> torch.Tensor:
        """
        Forward pass for flow stream encoding (fully batched).

        Args:
            dept_features: Normalized department properties [service_time, service_weight]
            flow_matrix: Normalized patient flow between departments
            dept_mask: Boolean mask for valid departments

        Returns:
            Flow embeddings of shape (batch, n_depts, hidden_dim)
        """
        adj_norm = self._build_normalized_adjacency(flow_matrix, dept_mask)
        mask_col = dept_mask.unsqueeze(-1).to(dept_features.dtype)  # (batch, n, 1)

        x = self.input_projection(dept_features) * mask_col

        for i in range(self.num_layers):
            identity = x
            x = torch.bmm(adj_norm, self.gcn_linears[i](x))  # propagate
            x = self.norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + identity
            x = x * mask_col  # keep padding rows at zero

        return self.output_projection(x) * mask_col


class CrossAttentionFusion(nn.Module):
    """
    Fuses physical and flow embeddings using cross-attention.

    The key insight is that we need to ALIGN the embeddings before fusion:
    - Physical embeddings are indexed by SLOT
    - Flow embeddings are indexed by DEPARTMENT
    - We use dept_to_slot mapping to align them

    After alignment, cross-attention allows:
    - Physical context to attend to flow demand (what departments need)
    - Flow context to attend to physical constraints (what's possible)

    Args:
        hidden_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim: int = hidden_dim

        self.cross_attn_phys_to_flow = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn_flow_to_phys = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_phys = nn.LayerNorm(hidden_dim)
        self.norm_flow = nn.LayerNorm(hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        h_phys: torch.Tensor,  # (batch, n_slots, hidden_dim) - slot-indexed
        h_flow: torch.Tensor,  # (batch, n_depts, hidden_dim) - dept-indexed
        dept_to_slot: torch.Tensor,  # (batch, n_depts) - layout mapping
        slot_to_dept: torch.Tensor,  # (batch, n_slots) - reverse mapping
        node_mask: torch.Tensor,  # (batch, n) - valid node mask
    ) -> torch.Tensor:
        """
        Fuses physical and flow embeddings.

        The critical step is ALIGNMENT:
        1. h_flow is indexed by dept_idx
        2. dept_to_slot[dept_idx] = slot_idx tells us where each dept is
        3. We need to reorder h_flow to match slot indexing

        Args:
            h_phys: Physical embeddings (slot-indexed)
            h_flow: Flow embeddings (dept-indexed)
            dept_to_slot: Current layout mapping
            slot_to_dept: Reverse layout mapping
            node_mask: Valid node mask

        Returns:
            Fused embeddings (slot-indexed) of shape (batch, n, hidden_dim)
        """
        batch, n, _ = h_phys.shape

        # We need h_flow_aligned[s] = h_flow[d] where dept_to_slot[d] = s
        # This means: slot s contains department d, so get flow info for d

        gather_idx = slot_to_dept.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )  # (batch, n_slots, hidden_dim)
        h_flow_aligned = torch.gather(
            h_flow, dim=1, index=gather_idx
        )  # (batch, n_slots, hidden_dim)

        key_padding_mask = ~node_mask

        h_phys_cross, _ = self.cross_attn_phys_to_flow(
            query=h_phys,
            key=h_flow_aligned,
            value=h_flow_aligned,
            key_padding_mask=key_padding_mask,
        )

        h_phys_enhanced = self.norm_phys(h_phys + h_phys_cross)

        h_flow_cross, _ = self.cross_attn_flow_to_phys(
            query=h_flow_aligned,
            key=h_phys,
            value=h_phys,
            key_padding_mask=key_padding_mask,
        )

        h_flow_enhanced = self.norm_flow(h_flow_aligned + h_flow_cross)

        h_concat = torch.cat([h_phys_enhanced, h_flow_enhanced], dim=-1)
        h_fused = self.fusion_mlp(h_concat)

        return h_fused  # (batch, n, hidden_dim)


class DualStreamGNNEncoder(nn.Module):
    """
    Complete dual-stream GNN encoder for hospital layout optimization.

    This encoder combines:
    1. PhysicalStreamEncoder: Processes slot features and distance relationships
    2. FlowStreamEncoder: Processes department features and flow patterns
    3. CrossAttentionFusion: Aligns and fuses both streams

    The output is a set of node embeddings that capture both:
    - Physical constraints (what layouts are feasible)
    - Flow requirements (what adjacencies are beneficial)

    Usage:
        encoder = DualStreamGNNEncoder(...)

        # During forward pass:
        node_embeddings = encoder(
            slot_features=slot_features,      # (batch, n, 4)
            distance_matrix=distance_matrix,  # (batch, n, n)
            dept_features=dept_features,      # (batch, n, 2)
            flow_matrix=flow_matrix,          # (batch, n, n)
            dept_to_slot=dept_to_slot,        # (batch, n)
            node_mask=node_mask,              # (batch, n)
        )

    Args:
        slot_feat_dim: Dimension of slot features (area, x, y, z)
        dept_feat_dim: Dimension of department features (service_time, service_weight)
        hidden_dim: Hidden dimension for all components
        output_dim: Output embedding dimension
        num_phys_layers: Number of layers in physical encoder
        num_flow_layers: Number of layers in flow encoder
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        slot_feat_dim: int = 4,
        dept_feat_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_phys_layers: int = 4,
        num_flow_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        self.phys_encoder = PhysicalStreamEncoder(
            slot_feat_dim=slot_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_phys_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.flow_encoder = FlowStreamEncoder(
            dept_feat_dim=dept_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        slot_features: torch.Tensor,  # (batch, n, slot_feat_dim)
        distance_matrix: torch.Tensor,  # (batch, n, n)
        dept_features: torch.Tensor,  # (batch, n, dept_feat_dim)
        flow_matrix: torch.Tensor,  # (batch, n, n)
        dept_to_slot: torch.Tensor,  # (batch, n)
        slot_to_dept: torch.Tensor,  # (batch, n)
        node_mask: torch.Tensor,  # (batch, n) - True for valid
    ) -> torch.Tensor:
        """
        Forward pass through the dual-stream encoder.

        Args:
            slot_features: Normalized slot properties [area, x, y, z]
            distance_matrix: Normalized slot-to-slot distances
            dept_features: Normalized department properties [service_time, weight]
            flow_matrix: Normalized department-to-department flow
            dept_to_slot: Current layout mapping
            slot_to_dept: Reverse layout mapping
            node_mask: Boolean mask for valid nodes

        Returns:
            Node embeddings of shape (batch, n, output_dim)
        """
        h_phys = self.phys_encoder(
            slot_features=slot_features,
            distance_matrix=distance_matrix,
            slot_mask=node_mask,
        )

        h_flow = self.flow_encoder(
            dept_features=dept_features,
            flow_matrix=flow_matrix,
            dept_mask=node_mask,
        )

        h_fused = self.fusion(
            h_phys=h_phys,
            h_flow=h_flow,
            dept_to_slot=dept_to_slot,
            slot_to_dept=slot_to_dept,
            node_mask=node_mask,
        )

        return self.output_projection(h_fused)

    def get_output_dim(self) -> int:
        return self.output_dim
