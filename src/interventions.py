# Copyright 2026 — jspace-replication project (Apache-2.0, matching upstream)
"""Residual-stream swap intervention, following the paper's convention:

    "Swap — clamping a lens coordinate replaces one token's direction with
    another's at every band layer at the specified positions."
    (jacobian-lens data/experiments/README.md)

Implementation choices (documented because upstream ships no intervention
code):

- A token's direction at layer l is the transported unembedding row
  ``d = normalize(J_l^T @ W_U[token])`` (cf. the "unit-normalized transpose
  row" wording in the verbal-introspection prompt-set description). ``W_U``
  is the raw output-embedding matrix; the final norm is scale-invariant, so
  normalization absorbs it.
- The swap transfers the source coordinate to the target direction at every
  hooked position:  ``h' = h - (h·dA) dA + (h·dA) dB``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def token_direction(lens, unembed_weight: torch.Tensor, token_id: int, layer: int) -> torch.Tensor:
    """normalize(J_l^T @ W_U[token]) in float32 on the weight's device."""
    u = unembed_weight[token_id].float()
    J = lens.jacobians[layer].to(u.device).float()
    d = J.T @ u
    return d / d.norm()


class SwapHooks:
    """Context manager: swap token A's direction for token B's at the output
    of every block in ``layers``, at all positions of the current forward.

    Args:
        blocks: model.layers (jlens LensModel convention).
        lens: fitted JacobianLens (provides J_l).
        unembed_weight: [vocab, d_model] output-embedding matrix.
        token_a / token_b: single token ids to swap (A -> B).
        layers: band layer indices; must be subset of lens.source_layers.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        lens,
        unembed_weight: torch.Tensor,
        token_a: int,
        token_b: int,
        layers: Sequence[int],
    ) -> None:
        unknown = set(layers) - set(lens.source_layers)
        if unknown:
            raise ValueError(f"layers {sorted(unknown)} not in lens.source_layers")
        self._blocks = blocks
        self._layers = list(layers)
        self._dirs = {
            l: (
                token_direction(lens, unembed_weight, token_a, l),
                token_direction(lens, unembed_weight, token_b, l),
            )
            for l in self._layers
        }
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int):
        d_a, d_b = self._dirs[layer]

        def hook(module: nn.Module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            h = tensor.float()
            coeff = h @ d_a  # [batch, seq]
            h = h - coeff.unsqueeze(-1) * d_a + coeff.unsqueeze(-1) * d_b
            new = h.to(tensor.dtype)
            if torch.is_tensor(output):
                return new
            return (new, *output[1:])

        return hook

    def __enter__(self) -> "SwapHooks":
        try:
            for layer in self._layers:
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
