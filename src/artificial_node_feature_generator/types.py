from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProviderOutput:
    features: torch.Tensor
    source: str
    cacheable: bool = False
