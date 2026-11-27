from __future__ import annotations

import torch
import torch.nn.functional as F


class FP32ScoringHead(torch.nn.Linear):
    """Keep the scoring projection in FP32, including under caller autocast.

    Defined at module level so multiprocessing (spawn) can pickle models that hold it.
    """

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=torch.float32,
        )
        self.load_state_dict(linear.state_dict())
        self.train(linear.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # a later model.half() / .to(bf16) also casts this head, so upcast here too
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(
                inputs.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float(),
            )
