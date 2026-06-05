"""
Autoregressive Actor and Critic for Hospital Layout Optimization (TorchRL Version)

This module implements the policy network for selecting department swap actions
using an autoregressive two-step selection process:

    Step 1: Select the first department based on node + global embeddings
    Step 2: Select the second department (excluding the first) with additional context

The key improvement over the Tianshou version is the incorporation of a global
graph embedding that provides context about the overall layout state, enabling
better action decisions.

Architecture:
    Node Embeddings (from DualStreamGNNEncoder)
            │
            ▼
    ┌───────────────┐
    │ Global Pool   │ ──► graph_embed (global context)
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ First Head    │ ──► action1, log_prob1
    │ [node, graph] │
    └───────────────┘
            │
            ▼
    ┌────────────────────┐
    │ Second Head        │ ──► action2, log_prob2
    │ [node, graph, a1]  │
    └────────────────────┘
            │
            ▼
    joint_log_prob = log_prob1 + log_prob2
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.distributions import Categorical

from .encoder import DualStreamGNNEncoder


@dataclass
class ActorOutput:
    action1: torch.Tensor
    action2: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    log_prob1: torch.Tensor
    log_prob2: torch.Tensor


@dataclass
class ActorCriticOutput:
    action1: torch.Tensor
    action2: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    log_prob1: torch.Tensor
    log_prob2: torch.Tensor
    value: torch.Tensor
    node_embeddings: torch.Tensor


class GlobalPooling(nn.Module):
    """
    Computes a global graph embedding from node embeddings.

    Supports multiple pooling strategies:
    - mean: Average pooling (default, stable gradients)
    - max: Max pooling (captures salient features)
    - attention: Learned attention weights (most expressive)
    """

    def __init__(
        self,
        hidden_dim: int,
        pooling_type: Literal['mean', 'max', 'attention'] = 'attention',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling_type: Literal['mean', 'max', 'attention'] = pooling_type
        self.hidden_dim: int = hidden_dim

        if pooling_type == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,  # (batch, n, hidden_dim)
    ) -> torch.Tensor:
        if self.pooling_type == 'mean':
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x

        elif self.pooling_type == 'max':
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(1).values
            return x

        elif self.pooling_type == 'attention':
            scores = self.attn(x).squeeze(-1)  # (batch, n)
            scores = scores.masked_fill(~mask, float('-inf'))
            weights = F.softmax(scores, dim=-1)
            x = torch.einsum('bn,bnd->bd', weights, x)  # (batch, hidden_dim)
            x = self.dropout(x)
            return x

        raise ValueError(f'Invalid pooling type: {self.pooling_type}')


class AutoregressiveActor(nn.Module):
    """
    Autoregressive policy network for selecting two departments to swap.

    The selection is performed in two steps:
    1. First department: Based on [node_embed, graph_embed]
    2. Second department: Based on [node_embed, graph_embed, first_node_embed]
       with the first selected node masked out

    This design captures:
    - Local context: What makes each department a good swap candidate
    - Global context: Overall layout state and optimization progress
    - Pairwise context: Relationship between the two selected departments

    Args:
        node_embed_dim: Dimension of node embeddings from encoder
        hidden_dim: Hidden dimension for action heads
        pooling_type: Type of global pooling ("mean", "max", "attention")
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        node_embed_dim: int,
        hidden_dim: int = 256,
        pooling_type: Literal['mean', 'max', 'attention'] = 'attention',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_embed_dim: int = node_embed_dim
        self.hidden_dim: int = hidden_dim

        self.global_pooling = GlobalPooling(
            hidden_dim=node_embed_dim,
            pooling_type=pooling_type,
            dropout=dropout,
        )

        # First action head: [node_embed, graph_embed] → logits
        # Input: node_embed (node_embed_dim) + graph_embed (node_embed_dim)
        self.first_action_head = nn.Sequential(
            nn.Linear(node_embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Second action head: [node_embed, graph_embed, first_node_embed] → logits
        # Input: node_embed + graph_embed + first_node_embed = 3 * node_embed_dim
        self.second_action_head = nn.Sequential(
            nn.Linear(node_embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _create_masked_distribution(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> Categorical:
        mask_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, mask_value)
        return Categorical(logits=masked_logits)

    def _get_dist1(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        swap_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        _, n, _ = node_embeddings.shape
        graph_embed = self.global_pooling(node_embeddings, node_mask)  # node_mask: pooling
        graph_embed_expanded = graph_embed.unsqueeze(1).expand(-1, n, -1)
        logits1 = self.first_action_head(
            torch.cat([node_embeddings, graph_embed_expanded], dim=-1)
        ).squeeze(-1)  # (batch, n)

        # action1 candidates: departments with at least one legal swap partner
        action1_mask = swap_mask.any(dim=-1)  # (batch, n)
        # graceful fallback to node_mask if a sample has no legal swap (avoid NaN)
        has_legal = action1_mask.any(dim=-1, keepdim=True)
        action1_mask = torch.where(has_legal, action1_mask, node_mask)
        return self._create_masked_distribution(
            logits1, action1_mask
        ), graph_embed_expanded

    def _get_dist2(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        graph_embed_expanded: torch.Tensor,
        swap_mask: torch.Tensor,
        action1: torch.Tensor,
    ) -> Categorical:
        batch, n, d = node_embeddings.shape

        gather_idx = action1.view(batch, 1, 1).expand(-1, -1, d)
        first_node = node_embeddings.gather(dim=1, index=gather_idx).squeeze(1)
        first_node_expanded = first_node.unsqueeze(1).expand(-1, n, -1)

        logits2 = self.second_action_head(
            torch.cat(
                [node_embeddings, graph_embed_expanded, first_node_expanded], dim=-1
            )
        ).squeeze(-1)  # (batch, n)

        # action2 candidates = swap_mask[action1] row; swap_mask[a, a] = False
        # already excludes action1 itself
        action2_mask = swap_mask.gather(
            dim=1, index=action1.view(batch, 1, 1).expand(-1, 1, n)
        ).squeeze(1)  # (batch, n)
        # graceful fallback to node_mask minus action1 if the row is empty
        has_legal = action2_mask.any(dim=-1, keepdim=True)
        fallback = node_mask.clone()
        fallback.scatter_(dim=1, index=action1.unsqueeze(-1), value=False)
        action2_mask = torch.where(has_legal, action2_mask, fallback)

        return self._create_masked_distribution(logits2, action2_mask)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        swap_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        """
        Forward pass: select two departments autoregressively.

        Args:
            node_embeddings: Node-level features from DualStreamGNNEncoder
            node_mask: Boolean mask indicating valid nodes (True = valid)
            swap_mask: (batch, n, n) bool, legal swap pairs (swappable + area)
            deterministic: If True, select argmax instead of sampling

        Returns:
            ActorOutput containing:
            - action1: First selected department index (batch,)
            - action2: Second selected department index (batch,)
            - log_prob: Joint log probability (batch)
            - entropy: Joint entropy for exploration bonus (batch)
            - log_prob1: Log prob of first action (batch)
            - log_prob2: Log prob of second action (batch)
        """
        dist1, graph_embed_expanded = self._get_dist1(
            node_embeddings, node_mask, swap_mask
        )
        action1 = dist1.probs.argmax(dim=-1) if deterministic else dist1.sample()  # type: ignore

        dist2 = self._get_dist2(
            node_embeddings, node_mask, graph_embed_expanded, swap_mask, action1
        )
        action2 = dist2.probs.argmax(dim=-1) if deterministic else dist2.sample()  # type: ignore

        log_prob1 = dist1.log_prob(action1)  # (batch,)
        log_prob2 = dist2.log_prob(action2)  # (batch,)
        entropy1 = dist1.entropy()  # (batch,)
        entropy2 = dist2.entropy()  # (batch,)
        log_prob = log_prob1 + log_prob2
        entropy = entropy1 + entropy2

        return ActorOutput(
            action1=action1,
            action2=action2,
            log_prob=log_prob,
            entropy=entropy,
            log_prob1=log_prob1,
            log_prob2=log_prob2,
        )

    @torch.no_grad()
    def get_action(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        swap_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions only.

        Returns:
            action1, action2: Selected department indices
        """
        result = self.forward(node_embeddings, node_mask, swap_mask, deterministic)
        return result.action1, result.action2

    def evaluate_actions(
        self,
        node_embeddings: torch.Tensor,  # (batch, n, hidden_dim)
        node_mask: torch.Tensor,  # (batch, n)
        swap_mask: torch.Tensor,  # (batch, n, n)
        action1: torch.Tensor,  # (batch,)
        action2: torch.Tensor,  # (batch,)
    ) -> ActorOutput:
        """
        Evaluate log probabilities and entropy for given actions.

        Used during PPO training to compute importance ratios.

        Args:
            node_embeddings: Node features from encoder
            node_mask: Valid node mask
            swap_mask: Legal swap pairs (swappable + area)
            action1: First action to evaluate
            action2: Second action to evaluate

        Returns:
            ActorOutput containing log_prob, entropy, log_prob1, log_prob2
        """

        dist1, graph_embed_expanded = self._get_dist1(
            node_embeddings, node_mask, swap_mask
        )
        dist2 = self._get_dist2(
            node_embeddings, node_mask, graph_embed_expanded, swap_mask, action1
        )

        log_prob1 = dist1.log_prob(action1)  # (batch,)
        log_prob2 = dist2.log_prob(action2)  # (batch,)
        entropy1 = dist1.entropy()  # (batch,)
        entropy2 = dist2.entropy()  # (batch,)
        log_prob = log_prob1 + log_prob2
        entropy = entropy1 + entropy2

        return ActorOutput(
            action1=action1,
            action2=action2,
            log_prob=log_prob,
            entropy=entropy,
            log_prob1=log_prob1,
            log_prob2=log_prob2,
        )


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    activation: type[nn.Module] = nn.ReLU,
    dropout: float = 0.1,
    norm_first_layer: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    in_dim = input_dim
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if norm_first_layer and i == 0:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    """
    Value network (Critic) for estimating state values.

    Takes node embeddings, pools them to a graph-level representation,
    and outputs a scalar value estimate.

    Args:
        node_embed_dim: Dimension of node embeddings
        hidden_dim: Hidden dimension for value MLP
        num_layers: Number of hidden layers
        pooling_type: Type of global pooling
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        pooling_type: Literal['mean', 'max', 'attention'] = 'attention',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.global_pooling = GlobalPooling(
            hidden_dim=node_embed_dim,
            pooling_type=pooling_type,
            dropout=dropout,
        )

        self.value_head = build_mlp(
            input_dim=node_embed_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=nn.ReLU,
            dropout=dropout,
            norm_first_layer=True,
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (batch, n, hidden_dim)
        node_mask: torch.Tensor,  # (batch, n)
    ) -> torch.Tensor:
        """
        Compute state value estimate.

        Returns:
            value: Scalar value for each batch element (batch,)
        """
        graph_embed = self.global_pooling(
            node_embeddings, node_mask
        )  # (batch, embed_dim)
        value = self.value_head(graph_embed).squeeze(-1)  # (batch,)
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module for PPO training.

    This wrapper combines:
    - DualStreamGNNEncoder: Encodes observations into node embeddings
    - AutoregressiveActor: Selects swap actions
    - ValueNet: Estimates state values

    For TorchRL integration, this can be wrapped with TensorDictModule.

    Args:
        encoder: Pre-initialized DualStreamGNNEncoder
        actor: Pre-initialized AutoregressiveActor
        critic: Pre-initialized ValueNet
    """

    def __init__(
        self,
        encoder: DualStreamGNNEncoder,
        actor: AutoregressiveActor,
        critic: ValueNet,
    ):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

    def forward(
        self,
        slot_features: torch.Tensor,  # (batch, n, 4)
        distance_matrix: torch.Tensor,  # (batch, n, n)
        dept_features: torch.Tensor,  # (batch, n, 2)
        flow_matrix: torch.Tensor,  # (batch, n, n)
        dept_to_slot: torch.Tensor,  # (batch, n)
        slot_to_dept: torch.Tensor,  # (batch, n)
        node_mask: torch.Tensor,  # (batch, n)
        swap_mask: torch.Tensor,  # (batch, n, n)
        deterministic: bool = False,
    ) -> ActorCriticOutput:
        """
        Full forward pass: encode → act → value.

        Returns dictionary with action1, action2, log_prob, entropy, value.
        """
        node_embeddings = self.encoder(
            slot_features=slot_features,
            distance_matrix=distance_matrix,
            dept_features=dept_features,
            flow_matrix=flow_matrix,
            dept_to_slot=dept_to_slot,
            slot_to_dept=slot_to_dept,
            node_mask=node_mask,
        )

        actor_output = self.actor(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            swap_mask=swap_mask,
            deterministic=deterministic,
        )

        value = self.critic(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
        )

        return ActorCriticOutput(
            action1=actor_output.action1,
            action2=actor_output.action2,
            log_prob=actor_output.log_prob,
            entropy=actor_output.entropy,
            log_prob1=actor_output.log_prob1,
            log_prob2=actor_output.log_prob2,
            value=value,
            node_embeddings=node_embeddings,
        )

    def get_value(
        self,
        slot_features: torch.Tensor,
        distance_matrix: torch.Tensor,
        dept_features: torch.Tensor,
        flow_matrix: torch.Tensor,
        dept_to_slot: torch.Tensor,
        slot_to_dept: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.encoder(
            slot_features=slot_features,
            distance_matrix=distance_matrix,
            dept_features=dept_features,
            flow_matrix=flow_matrix,
            dept_to_slot=dept_to_slot,
            slot_to_dept=slot_to_dept,
            node_mask=node_mask,
        )
        return self.critic(node_embeddings=node_embeddings, node_mask=node_mask)

    def evaluate_actions(
        self,
        slot_features: torch.Tensor,
        distance_matrix: torch.Tensor,
        dept_features: torch.Tensor,
        flow_matrix: torch.Tensor,
        dept_to_slot: torch.Tensor,
        slot_to_dept: torch.Tensor,
        node_mask: torch.Tensor,
        swap_mask: torch.Tensor,
        action1: torch.Tensor,
        action2: torch.Tensor,
    ) -> ActorCriticOutput:
        node_embeddings = self.encoder(
            slot_features=slot_features,
            distance_matrix=distance_matrix,
            dept_features=dept_features,
            flow_matrix=flow_matrix,
            dept_to_slot=dept_to_slot,
            slot_to_dept=slot_to_dept,
            node_mask=node_mask,
        )

        actor_eval = self.actor.evaluate_actions(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            swap_mask=swap_mask,
            action1=action1,
            action2=action2,
        )
        value = self.critic(node_embeddings=node_embeddings, node_mask=node_mask)

        return ActorCriticOutput(
            action1=actor_eval.action1,
            action2=actor_eval.action2,
            log_prob=actor_eval.log_prob,
            entropy=actor_eval.entropy,
            log_prob1=actor_eval.log_prob1,
            log_prob2=actor_eval.log_prob2,
            value=value,
            node_embeddings=node_embeddings,
        )


def create_actor_critic(
    encoder: DualStreamGNNEncoder,
    node_embed_dim: int = 256,
    actor_hidden_dim: int = 256,
    critic_hidden_dim: int = 256,
    critic_num_layers: int = 2,
    pooling_type: Literal['mean', 'max', 'attention'] = 'attention',
    dropout: float = 0.1,
) -> ActorCritic:
    """
    Factory function to create a complete ActorCritic module.

    Args:
        encoder: Pre-initialized DualStreamGNNEncoder
        node_embed_dim: Output dimension of encoder (must match encoder.output_dim)
        actor_hidden_dim: Hidden dimension for actor heads
        critic_hidden_dim: Hidden dimension for value network
        critic_num_layers: Number of layers in value network
        pooling_type: Global pooling strategy
        dropout: Dropout rate

    Returns:
        Initialized ActorCritic module
    """
    actor = AutoregressiveActor(
        node_embed_dim=node_embed_dim,
        hidden_dim=actor_hidden_dim,
        pooling_type=pooling_type,
        dropout=dropout,
    )
    critic = ValueNet(
        node_embed_dim=node_embed_dim,
        hidden_dim=critic_hidden_dim,
        num_layers=critic_num_layers,
        pooling_type=pooling_type,
        dropout=dropout,
    )
    return ActorCritic(encoder=encoder, actor=actor, critic=critic)
