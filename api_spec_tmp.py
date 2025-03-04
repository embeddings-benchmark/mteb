from __future__ import annotations

import numpy as np
import torch

arr = np.array([[1, 2], [1, 2]])

# ideally we should use expand_dims (torch does not implement this -- issue:
# unsqueeze? - does this actually do changes in mem.?
arr.reshape(1, *arr.shape).shape


arr = torch.Tensor(arr)
torch.mm(arr, arr) == arr @ arr  # eq.
torch.mm(arr, arr) == arr.matmul(arr)  # also eq.

# https://stackoverflow.com/questions/73924697/whats-the-difference-between-torch-mm-torch-matmul-and-torch-mul
# torch.mm does not broadcast, @ (matmul) does


# Search replaces:
# arr.unsqueeze(0)
# torch.tensor(),  torch.from_numpy, numpy.array() (and other types of tensor conversions)

# Speed:
# measure speed before and after
# measure speed difference between torch and numpy

# problems
# 1: Unsqueeze not in array API spec, we should use expand_dims, but torch does not implement this (issue). We can use reshape instead.
# 2: What to do with torch.functional? (we would have to reimplent these to make it work)
#
# future problems:
# 1: Allow for GPU during evaluation
