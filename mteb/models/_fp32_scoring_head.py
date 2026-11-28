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


class TokenLogitUpcast:
    """Forward hook recomputing the given vocabulary logits of a low-precision lm_head in fp32.

    A module-level class so spawn can pickle the hook along with the model.
    """

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def __call__(
        self,
        module: torch.nn.Linear,
        args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = self.token_ids
        with torch.autocast(device_type=output.device.type, enabled=False):
            bias = None if module.bias is None else module.bias[token_ids].float()
            rows = F.linear(args[0].float(), module.weight[token_ids].float(), bias)
        output = output.float()
        output[..., token_ids] = rows
        return output
