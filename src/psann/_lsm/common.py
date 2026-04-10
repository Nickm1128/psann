from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]


def _to_float_tensor(
    data: TensorLike,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, bool, torch.device, torch.dtype]:
    if isinstance(data, torch.Tensor):
        original_device = data.device
        original_dtype = data.dtype
        tensor = data.detach()
        if tensor.device != device or tensor.dtype != torch.float32:
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device)
        return tensor.contiguous(), True, original_device, original_dtype
    arr = np.asarray(data, dtype=np.float32)
    tensor = torch.from_numpy(arr).to(device)
    return tensor.contiguous(), False, torch.device("cpu"), torch.float32


def _tensor_to_output(
    tensor: torch.Tensor,
    *,
    return_tensor: bool,
    target_device: torch.device,
    target_dtype: torch.dtype,
):
    result = tensor.detach()
    if return_tensor:
        return result.to(device=target_device, dtype=target_dtype)
    return result.cpu().numpy()
