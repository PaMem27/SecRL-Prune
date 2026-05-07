# pruned_view.py (simple)
# Minimal masking view for LLaMA-like MLPs: gate_proj, up_proj, down_proj.
# - Discovers mlp modules under model.model.layers[i].mlp
# - Patches mlp.forward to insert a channel mask after SiLU(gate)*up
# - Masks are per-layer boolean tensors of shape [hidden_dim] (intermediate size)
# - No weight slicing/pruning here; purely a runtime mask for training/eval

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunedMLPView:
    def __init__(self, model: nn.Module):
        self.model = model
        self.mlp_modules: List[nn.Module] = []
        self.hidden_sizes: List[int] = []
        self._orig_forwards: Dict[int, types.FunctionType] = {}
        self._attached: bool = False
        # current masks (device tensors shaped [H]) or None
        self._masks: Dict[int, Optional[torch.Tensor]] = {}

        self._discover_llama_mlps()

    # --- discovery ---
    def _discover_llama_mlps(self) -> None:
        """
        Very simple discovery for HF LLaMA/CodeLLaMA:
          model.model.layers: list of blocks, each has .mlp with gate_proj/up_proj/down_proj
        """
        root = getattr(self.model, "model", None)
        layers = getattr(root, "layers", None)
        if not isinstance(layers, (list, nn.ModuleList)):
            raise RuntimeError("Could not find model.model.layers for LLaMA-like model.")

        self.mlp_modules.clear()
        self.hidden_sizes.clear()
        for i, blk in enumerate(layers):
            mlp = getattr(blk, "mlp", None)
            if mlp is None:
                continue
            if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj")):
                continue
            # Infer intermediate size from gate_proj weight: [H, in_dim]
            w = mlp.gate_proj.weight
            H = w.shape[0]
            self.mlp_modules.append(mlp)
            self.hidden_sizes.append(H)
            self._masks[i] = None  # default: no pruning

        if not self.mlp_modules:
            raise RuntimeError("No LLaMA-style MLPs discovered (gate_proj/up_proj/down_proj).")

    # --- attach/detach ---
    def attach(self) -> None:
        if self._attached:
            return
        for idx, mlp in enumerate(self.mlp_modules):
            # Save and replace forward
            self._orig_forwards[idx] = mlp.forward

            def make_forward(idx_: int, mlp_: nn.Module):
                def forward(x: torch.Tensor) -> torch.Tensor:
                    # Re-implement simple LLaMA MLP: SiLU(gate_proj(x)) * up_proj(x) -> down_proj(...)
                    g = mlp_.gate_proj(x)
                    u = mlp_.up_proj(x)
                    h = F.silu(g) * u  # [B, H]
                    m = self._masks.get(idx_)
                    if m is not None:
                        # ensure device/dtype, broadcast over batch/time dims
                        if m.device != h.device:
                            m_local = m.to(h.device)
                        else:
                            m_local = m
                        h = h * m_local.unsqueeze(0)  # [1, H]
                    y = mlp_.down_proj(h)
                    return y
                return forward

            mlp.forward = make_forward(idx, mlp)  # type: ignore[attr-defined]
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        for idx, mlp in enumerate(self.mlp_modules):
            if idx in self._orig_forwards:
                mlp.forward = self._orig_forwards[idx]  # type: ignore[attr-defined]
        self._orig_forwards.clear()
        self._attached = False

    # --- mask control ---
    @torch.no_grad()
    def clear_all_masks(self) -> None:
        for idx in range(len(self.mlp_modules)):
            self._masks[idx] = None

    @torch.no_grad()
    def set_global_masks(self, mask_dict: Dict[int, torch.Tensor]) -> None:
        """
        mask_dict: maps layer index (0..L-1) to a boolean/0-1 tensor [H].
        Missing indices leave mask as-is (typically None = no pruning).
        """
        for idx, mask in mask_dict.items():
            if idx < 0 or idx >= len(self.mlp_modules):
                continue
            mask = mask.to(dtype=self.mlp_modules[idx].gate_proj.weight.dtype, device=self.mlp_modules[idx].gate_proj.weight.device)
            self._masks[idx] = mask

    @torch.no_grad()
    def export_current_bool_masks(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for idx, H in enumerate(self.hidden_sizes):
            m = self._masks.get(idx)
            if m is None:
                out.append(torch.ones(H, dtype=torch.bool, device="cpu"))
            else:
                out.append((m.detach().to("cpu") > 0.5))
        return out