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
- ``DirectionSwapHooks`` applies the same transfer along arbitrary
  precomputed unit directions (e.g. linear-probe / mass-mean directions),
  so the direction source can be varied while the mechanism stays fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def token_direction(lens, unembed_weight: torch.Tensor, token_id: int, layer: int) -> torch.Tensor:
    """normalize(J_l^T @ W_U[token]) in float32 on the weight's device."""
    u = unembed_weight[token_id].float()
    J = lens.jacobians[layer].to(u.device).float()
    d = J.T @ u
    return d / d.norm()


class DirectionSwapHooks:
    """Context manager: at the output of every block in ``dirs``, transfer
    the coordinate along direction A to direction B at all positions of the
    current forward.

    Args:
        blocks: model.layers (jlens LensModel convention).
        dirs: {layer index: (d_a, d_b)}, unit float32 vectors on the model
            device.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        dirs: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self._blocks = blocks
        self._dirs = dict(dirs)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # mean |h . d_a| over positions at each hooked layer, refreshed every
        # forward: the amplitude the swap harvests from the source direction.
        self.coeff_abs: dict[int, float] = {}

    def _make_hook(self, layer: int):
        d_a, d_b = self._dirs[layer]

        def hook(module: nn.Module, inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            h = tensor.float()
            coeff = h @ d_a  # [batch, seq]
            self.coeff_abs[layer] = float(coeff.abs().mean())
            h = h - coeff.unsqueeze(-1) * d_a + coeff.unsqueeze(-1) * d_b
            new = h.to(tensor.dtype)
            if torch.is_tensor(output):
                return new
            return (new, *output[1:])

        return hook

    def __enter__(self) -> "DirectionSwapHooks":
        try:
            for layer in self._dirs:
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


class SwapHooks(DirectionSwapHooks):
    """Swap token A's lens direction for token B's across ``layers``.

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
        dirs = {
            l: (
                token_direction(lens, unembed_weight, token_a, l),
                token_direction(lens, unembed_weight, token_b, l),
            )
            for l in layers
        }
        super().__init__(blocks, dirs)
