from __future__ import annotations

import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
# Try to use the problematic NMS function
try:
    from torchvision.ops import nms

    boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    idxs = nms(boxes, scores, 0.5)
    print("NMS operation successful!")
except Exception as e:
    print(f"Error: {e}")
