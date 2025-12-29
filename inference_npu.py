# inference_npu.py
"""
NPU inference backend (QNN / Snapdragon)

This file mirrors inference_cpu.py exactly at the interface level.
Actual QNN wiring will be added in Semester 2.
"""

from typing import Any
import torch


class NPUNotReadyError(RuntimeError):
    pass


def load_model(
    onnx_or_qnn_path: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Placeholder loader for NPU engine.

    Later this will:
      - load ONNX model
      - or load precompiled QNN context (.bin/.json)
    """
    engine = {
        "backend": "npu",
        "artifact": onnx_or_qnn_path,
    }
    return engine


def run(engine: dict, model_input_tensor: torch.Tensor) -> str:
    """
    Run inference on NPU.

    model_input_tensor:
        (T, C, H, W) float32 in [0,1]
    """
    if engine.get("backend") != "npu":
        raise NPUNotReadyError("Invalid engine passed to NPU backend.")

    # Explicit guard so you never accidentally call this
    raise NPUNotReadyError(
        "NPU backend not implemented yet.\n"
        "CPU path is confirmed working.\n"
        "Next step: export ONNX → compile with QNN → implement this function."
    )
