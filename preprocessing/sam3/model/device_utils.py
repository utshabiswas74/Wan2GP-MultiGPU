from contextlib import nullcontext
import os

import torch


def first_floating_parameter_dtype(model, default=torch.float32):
    if isinstance(model, torch.nn.Module):
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    return default


def cast_floating_tensors(value, dtype):
    if torch.is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() and value.dtype != dtype else value
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, dtype) for item in value)
    return value


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_accelerator_device() -> torch.device:
    if os.environ.get("WAN_MANUAL_PIPELINE_PARALLEL") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_accelerator_device(device) -> bool:
    if device is None:
        return False
    return torch.device(device).type in {"cuda", "mps"}


def accelerator_autocast(dtype=torch.bfloat16):
    device_type = get_accelerator_device().type
    if device_type in {"cuda", "mps"}:
        return torch.autocast(device_type=device_type, dtype=dtype)
    return nullcontext()


def empty_accelerator_cache():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    elif mps_is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
