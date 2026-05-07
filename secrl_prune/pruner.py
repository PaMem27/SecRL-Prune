# prunenet/pruner_policy.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# type-only import; safe if you don't have this file
if TYPE_CHECKING:
    from secrl_prune.pruned_view import PrunedMLPView


def _logit(p: float, eps: float = 1e-6) -> float:
    """Convert probability in (0,1) to logit safely."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p) - math.log(1.0 - p)


class _LayerSpec:
    """Minimal handle for an MLP block and its intermediate width."""
    def __init__(self, layer_idx: int, mlp_module: nn.Module, inter_dim: int):
        self.layer_idx = int(layer_idx)
        self.mlp = mlp_module
        self.inter_dim = int(inter_dim)


def _discover_mlp_layers(model: nn.Module) -> List[_LayerSpec]:
    """
    Minimal layer discovery for LLaMA/CodeLLaMA-style models:
      model.model.layers[i].mlp.down_proj
    """
    specs: List[_LayerSpec] = []
    root = getattr(model, "model", None) or getattr(model, "transformer", None)
    layers = getattr(root, "layers", None)
    if not isinstance(layers, (list, nn.ModuleList)):
        raise RuntimeError("Could not find model.model.layers (LLaMA-style).")

    for i, layer in enumerate(layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj") and isinstance(layer.mlp.down_proj, nn.Linear):
            inter_dim = int(layer.mlp.down_proj.in_features)
            specs.append(_LayerSpec(i, layer.mlp, inter_dim))

    if not specs:
        raise RuntimeError("No MLP blocks with down_proj found.")
    return specs


class MLPPrunerPolicy(nn.Module):
    """
    Minimal policy (PruneNet-style sampling):
      - Per-layer, per-group learnable logits (Parameters).
      - Convert logits to a categorical distribution over groups via softmax.
      - Sample a fixed number of groups per layer WITHOUT replacement using
        `torch.multinomial` (weighted sampling), analogous to PruneNet.
      - Log-prob is computed as sum of log-probs of the selected groups
        (same simplification PruneNet uses).
      - Returns unit masks, total logprob (for REINFORCE), entropy (categorical), and keep ratio.
    """
    def __init__(
        self,
        model: nn.Module,
        group_size: int = 16,
        init_keep_prob: float = 0.8,
        device: Optional[torch.device] = None,
        *,
        keep_target: Optional[float] = None,  # accepted but unused (kept for API compatibility)
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError("group_size must be > 0")

        self.device = device if device is not None else torch.device("cpu")
        self.group_size = int(group_size)

        # Discover MLP layers
        self.layer_specs: List[_LayerSpec] = _discover_mlp_layers(model)

        # Build per-layer, per-group logits as learnable parameters
        self.group_logits = nn.ParameterList()
        for spec in self.layer_specs:
            inter = spec.inter_dim
            G = (inter + self.group_size - 1) // self.group_size  # ceil
            init = torch.full((G,), _logit(float(init_keep_prob)), dtype=torch.float32)
            self.group_logits.append(nn.Parameter(init))

        # Default fraction of groups to keep if not specified per-call
        self.default_keep_frac = float(keep_target) if keep_target is not None else float(init_keep_prob)

        self.to(self.device)

    @torch.no_grad()
    def _expand_group_mask(self, group_mask: torch.Tensor, inter_dim: int) -> torch.Tensor:
        """Expand [G] group mask (0/1) to [inter_dim] unit mask (0/1)."""
        out = torch.zeros(inter_dim, dtype=torch.float32, device=group_mask.device)
        G = group_mask.numel()
        for g in range(G):
            start = g * self.group_size
            end = min(start + self.group_size, inter_dim)
            if start >= inter_dim:
                break
            out[start:end] = group_mask[g]
        return out

    def sample_masks(
        self,
        training: bool = True,
        threshold: float = 0.5,
        *,
        fixed_keep: Optional[float] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """
        PruneNet-style: per layer, sample exactly K groups without replacement.
        K = round((fixed_keep or default_keep_frac) * num_groups).

        Returns:
          mask_dict: {layer_idx: [inter_dim] float32 {0,1}}
          sum_logprob: scalar fp32 (REINFORCE objective)
          entropy: scalar fp32 (sum of categorical entropies per layer)
          keep_ratio: float in [0,1] over units
          layer_logprobs: [num_layers] tensor of per-layer logprob contributions
        """
        mask_dict: Dict[int, torch.Tensor] = {}
        sum_logprob = torch.zeros((), dtype=torch.float32, device=self.device)
        sum_entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        layer_logprobs_list: List[torch.Tensor] = []

        total_kept_units = 0.0
        total_units = 0

        keep_frac = float(self.default_keep_frac if fixed_keep is None else fixed_keep)
        keep_frac = max(0.0, min(1.0, keep_frac))

        for li, spec in enumerate(self.layer_specs):
            logits = self.group_logits[li].to(self.device)        # [G]
            weights = F.softmax(logits, dim=-1)                   # [G], positive, sums to 1
            weights = torch.clamp(weights, min=1e-12)             # numerical safety

            G = weights.numel()
            K = int(round(keep_frac * G))
            K = max(0, min(G, K))

            if training:
                if K == 0:
                    chosen = torch.empty((0,), dtype=torch.long, device=self.device)
                    logprob = torch.zeros((), dtype=torch.float32, device=self.device)
                else:
                    chosen = torch.multinomial(weights, num_samples=K, replacement=False)  # [K]
                    logprob = torch.log(weights.gather(0, chosen)).sum().float()
                actions = torch.zeros_like(weights, dtype=torch.float32)
                if K > 0:
                    actions.scatter_(0, chosen, 1.0)
                # Categorical entropy (not exact for k-of-n sampling, but useful as a signal)
                entropy = (-weights * torch.log(weights)).sum().float()
                layer_logprobs_list.append(logprob)
            else:
                # Deterministic: take top-K groups by weight
                if K == 0:
                    actions = torch.zeros_like(weights, dtype=torch.float32)
                else:
                    topk = torch.topk(weights, k=K, dim=0).indices
                    actions = torch.zeros_like(weights, dtype=torch.float32)
                    actions.scatter_(0, topk, 1.0)
                logprob = torch.zeros((), dtype=torch.float32, device=self.device)
                entropy = (-weights * torch.log(weights)).sum().float()
                layer_logprobs_list.append(torch.zeros((), dtype=torch.float32, device=self.device))

            sum_logprob = sum_logprob + logprob
            sum_entropy = sum_entropy + entropy

            unit_mask = self._expand_group_mask(actions, spec.inter_dim)  # [inter_dim]
            mask_dict[spec.layer_idx] = unit_mask

            total_kept_units += float(unit_mask.sum().item())
            total_units += int(spec.inter_dim)

        keep_ratio = total_kept_units / max(1, total_units)
        layer_logprobs = (
            torch.stack(layer_logprobs_list) if layer_logprobs_list else torch.zeros((0,), dtype=torch.float32, device=self.device)
        )
        return mask_dict, sum_logprob, sum_entropy, keep_ratio, layer_logprobs

    # Alias used by some code paths
    def forward(
        self,
        training: bool = True,
        threshold: float = 0.5,
        *,
        fixed_keep: Optional[float] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor, float, torch.Tensor]:
        return self.sample_masks(training=training, threshold=threshold, fixed_keep=fixed_keep)

    @torch.no_grad()
    def apply_to_view(self, view: "PrunedMLPView", mask_dict: Dict[int, torch.Tensor]) -> None:
        """Push masks into a PrunedMLPView instance."""
        view.set_global_masks(mask_dict)

    @torch.no_grad()
    def get_keep_probs(self) -> Dict[int, torch.Tensor]:
        """Per-layer Bernoulli keep probabilities over groups (not expanded to units)."""
        out: Dict[int, torch.Tensor] = {}
        for li, logits in enumerate(self.group_logits):
            out[li] = torch.sigmoid(logits.to(self.device))
        return out
