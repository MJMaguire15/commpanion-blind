# inference_cpu.py
import os
import torch
from types import SimpleNamespace

from auto_avsr.datamodule.data_module import DataModule
from auto_avsr.datamodule.transforms import TextTransform

# Use ONLY vendored ESPNet
from auto_avsr.espnet.nets.batch_beam_search import BatchBeamSearch
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from auto_avsr.espnet.nets.scorers.length_bonus import LengthBonus


# ---------------------------------------------------------
# Beam search helper
# ---------------------------------------------------------
def _get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    sos = model.odim - 1
    eos = model.odim - 1

    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


# ---------------------------------------------------------
# Model loader
# ---------------------------------------------------------
def load_model(ckpt_path: str, modality: str = "video", ctc_weight: float = 0.1):
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    torch.set_num_interop_threads(1)

    text_transform = TextTransform()
    token_list = text_transform.token_list

    model = E2E(
        len(token_list),
        modality,
        ctc_weight=ctc_weight,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    beam = _get_beam_search_decoder(model, token_list, ctc_weight=ctc_weight)

    return {
        "model": model,
        "beam": beam,
        "text": text_transform,
    }


# ---------------------------------------------------------
# Inference (CPU)
# ---------------------------------------------------------
@torch.no_grad()
def run(engine, model_input_tensor: torch.Tensor) -> str:
    """
    Expected input from DataModule:
        (T, C, H, W)   where C == 1
    ESPNet video frontend expects:
        (B, T, C, H, W)
    """
    model = engine["model"]
    beam = engine["beam"]
    text_transform = engine["text"]

    print("[DEBUG] raw input shape:", tuple(model_input_tensor.shape))

    # Handle accidental batching
    if model_input_tensor.dim() == 5:
        model_input_tensor = model_input_tensor[0]
        print("[DEBUG] took batch[0]:", tuple(model_input_tensor.shape))

    # Must be (T, C, H, W)
    if model_input_tensor.dim() != 4:
        raise RuntimeError(
            f"Expected (T,C,H,W), got {tuple(model_input_tensor.shape)}"
        )

    # Convert to float32
    xs = model_input_tensor.float()

    # Normalize if uint8
    if xs.max() > 1.0:
        xs = xs / 255.0

    # -------------------------------------------------
    # CRITICAL: ESPNet expects (B, T, C, H, W)
    # -------------------------------------------------
    xs = xs.unsqueeze(0)  # (1, T, C, H, W)

    print("[DEBUG] frontend input shape:", tuple(xs.shape))
    # EXPECTED: (1, T, 1, 88, 88)

    assert xs.shape[2] == 1, f"Expected C==1, got {xs.shape}"


    # Forward through ESPNet
    x = model.frontend(xs)
    x = model.proj_encoder(x)
    enc_feat, _ = model.encoder(x, None)
    enc_feat = enc_feat.squeeze(0)

    # Beam search decode
    nbest = beam(enc_feat)
    yseq = nbest[0].yseq[1:]  # drop <sos>
    token_ids = torch.tensor(list(map(int, yseq)))

    pred_text = text_transform.post_process(token_ids)
    return pred_text.replace("<eos>", "").strip()


# ---------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------
def run_from_roi_root(
    roi_root: str,
    test_file: str,
    ckpt_path: str,
    ctc_weight: float = 0.1,
) -> str:
    args = SimpleNamespace(
        modality="video",
        root_dir=str(roi_root),
        test_file=str(test_file),
        pretrained_model_path=str(ckpt_path),
        ctc_weight=float(ctc_weight),
        decode_snr_target=999999,
        debug=False,
    )

    dm = DataModule(args)
    dm.setup(stage="test")
    dl = dm.test_dataloader()

    batch = next(iter(dl))
    if not isinstance(batch, dict) or "input" not in batch:
        raise RuntimeError(f"Unexpected batch format: {type(batch)}")

    inp = batch["input"]

    engine = load_model(ckpt_path, modality="video", ctc_weight=ctc_weight)
    return run(engine, inp)
