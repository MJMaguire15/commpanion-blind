"""
AV-HuBERT runner (visual-only) skeleton.
Expose: decode(window: np.ndarray) -> str
- window: [T, H, W] float32 in [-1, 1], grayscale mouth ROI frames
Return: decoded string.
"""
import os
import numpy as np

_task = None
_model = None
_generator = None

CKPT_PATH = os.environ.get("AVHUBERT_CKPT", "models/avhubert/lrs3_vsr_large.pt")

def _lazy_init():
    global _task, _model, _generator
    if _model is not None:
        return
    try:
        import torch
        from fairseq import tasks, checkpoint_utils
        from fairseq.sequence_generator import SequenceGenerator
    except Exception as e:
        raise RuntimeError("av_hubert/fairseq not installed") from e

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"AV-HuBERT checkpoint not found at {CKPT_PATH}. "
                                f"Set AVHUBERT_CKPT env var to point to your .pt file.")

    overrides = {"modalities": ["video"]}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        filenames=[CKPT_PATH],
        arg_overrides=overrides,
    )
    model = models[0].eval()
    generator = SequenceGenerator([model], beam_size=20, len_penalty=1.0)

    _task = task
    _model = model
    _generator = generator

def decode(window: np.ndarray) -> str:
    _lazy_init()
    import torch

    x = torch.from_numpy(window).unsqueeze(0).unsqueeze(2).float()  # [1, T, 1, H, W]
    sample = {"net_input": {"source": x, "padding_mask": torch.zeros(x.shape[:2], dtype=torch.bool)}}
    hypos = _generator.generate([_model], sample)
    if not hypos or not hypos[0]:
        return ""
    tokens = hypos[0][0]["tokens"]
    text = _task.tgt_dict.string(tokens).strip()
    return text
