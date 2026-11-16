"""Central CPU/CUDA selection and checkpoint device helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def normalize_device_spec(value: str | torch.device | None) -> str:
    """Return a canonical CLI device specification without probing hardware."""

    if isinstance(value, torch.device):
        value = str(value)
    normalized = "auto" if value is None else str(value).strip().lower()
    if normalized in {"auto", "cpu", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        index = normalized.removeprefix("cuda:")
        if index.isdigit():
            return f"cuda:{int(index)}"
    raise ValueError(
        "Device must be one of: auto, cpu, cuda, or cuda:<nonnegative index>."
    )


def resolve_torch_device(
        value: str | torch.device | None = "auto",
) -> torch.device:
    """Resolve and validate an explicit CPU/CUDA request."""

    requested = normalize_device_spec(value)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {requested!r} was requested, but CUDA is unavailable."
        )
    if requested == "cuda":
        return torch.device("cuda")
    index = int(requested.split(":", maxsplit=1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(
            f"Device {requested!r} was requested, but only {count} CUDA "
            f"device(s) are visible."
        )
    return torch.device(requested)


def move_optimizer_state(
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> None:
    """Move tensors restored into an optimizer to its model's device."""

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, Mapping):
            return type(value)((key, move(item)) for key, item in value.items())
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            state[key] = move(value)


def cuda_rng_state() -> list[torch.Tensor] | None:
    """Capture all CUDA RNG streams when CUDA exists."""

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state_all()


def restore_cuda_rng_state(
        states: Sequence[torch.Tensor] | None,
        device: torch.device,
) -> None:
    """Restore the selected CUDA stream when a checkpoint contains it."""

    if device.type != "cuda" or not states or not torch.cuda.is_available():
        return
    target_index = (
        torch.cuda.current_device()
        if device.index is None
        else device.index
    )
    source_index = min(target_index, len(states) - 1)
    torch.cuda.set_rng_state(states[source_index].cpu(), device=target_index)

