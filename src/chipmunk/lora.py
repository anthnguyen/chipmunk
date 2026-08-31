"""Minimal LoRA with per-layer enable/disable.

Hand-rolled rather than PEFT because the layer-window experiment (PROTOCOL §6.3)
needs to activate the adapter on an arbitrary subset of blocks and measure the
behavioural effect. That is one flag here and fiddly in PEFT.

Convention: `layer i` means decoder block index i (0-based), matching
model.model.layers[i]. This differs from the residual-stream indexing used in
genabl (where layer l = hidden_states[l] = output of block l-1); the capture
code converts.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(x)), with a per-module on/off switch."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B stays zero: adapter starts as identity
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        return out + self.dropout(x) @ self.A.T @ self.B.T * self.scaling


def inject(model, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
           targets: tuple[str, ...] = DEFAULT_TARGETS,
           layers: list[int] | None = None) -> dict[str, LoRALinear]:
    """Replace target nn.Linear modules with LoRALinear. Returns {name: module}.

    `layers` restricts injection to those block indices (the "routed" arm of
    PROTOCOL §6.2). None means every block.
    """
    blocks = model.model.layers
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise RuntimeError(
            "this model already has LoRA adapters injected. Reuse the existing "
            "adapter dict (pass it to train(adapters=...)) or load a fresh model.")
    adapters: dict[str, LoRALinear] = {}
    for li, block in enumerate(blocks):
        if layers is not None and li not in layers:
            continue
        for name, module in list(block.named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in targets or not isinstance(module, nn.Linear):
                continue
            parent = block.get_submodule(name.rsplit(".", 1)[0]) if "." in name else block
            wrapped = LoRALinear(module, r, alpha, dropout)
            setattr(parent, leaf, wrapped)
            adapters[f"layers.{li}.{name}"] = wrapped
    if not adapters:
        raise RuntimeError(f"no modules matched targets={targets}")
    return adapters


def layer_of(name: str) -> int:
    return int(name.split(".")[1])


def set_enabled(adapters: dict[str, LoRALinear], layers: list[int] | None) -> None:
    """Enable the adapter only on `layers` (None = all on, [] = all off)."""
    for name, mod in adapters.items():
        mod.enabled = True if layers is None else (layer_of(name) in layers)


@contextmanager
def only_layers(adapters: dict[str, LoRALinear], layers: list[int] | None):
    """Temporarily activate the adapter on a subset of blocks.

    This is the layer-window sweep: the minimum contiguous window in which the
    behaviour still appears localises where the fine-tune's effect is *needed*,
    which per-layer ||delta|| cannot tell you (the residual stream carries an
    early change into every later layer).
    """
    prev = {n: m.enabled for n, m in adapters.items()}
    try:
        set_enabled(adapters, layers)
        yield
    finally:
        for n, m in adapters.items():
            m.enabled = prev[n]


@contextmanager
def disabled(adapters: dict[str, LoRALinear]):
    """Run as the unmodified base model. Used for h_base in the same process,
    on the same input tokens, so h_organism - h_base is a matched difference."""
    with only_layers(adapters, []):
        yield


def trainable_parameters(adapters: dict[str, LoRALinear]):
    for m in adapters.values():
        yield m.A
        yield m.B


def state_dict(adapters: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in adapters.items():
        out[f"{name}.A"] = m.A.detach().cpu()
        out[f"{name}.B"] = m.B.detach().cpu()
    return out


def load_state_dict(adapters: dict[str, LoRALinear], sd: dict[str, torch.Tensor]) -> None:
    for name, m in adapters.items():
        with torch.no_grad():
            m.A.copy_(sd[f"{name}.A"].to(m.A.device, m.A.dtype))
            m.B.copy_(sd[f"{name}.B"].to(m.B.device, m.B.dtype))


def effective_update_norm(adapters: dict[str, LoRALinear]) -> dict[int, float]:
    """Frobenius norm of the effective weight delta (scaling * B @ A) per block.

    A descriptive summary of where the fine-tune wrote, in *parameter* space.
    Not a substitute for the causal layer-window sweep, and not the same as the
    activation-space rank -- a low-rank parameter update can produce a
    higher-rank activation change (PROTOCOL §6.3).
    """
    per: dict[int, float] = {}
    for name, m in adapters.items():
        d = (m.B @ m.A) * m.scaling
        per[layer_of(name)] = per.get(layer_of(name), 0.0) + float(d.float().norm())
    return per
