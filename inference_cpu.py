# inference_cpu.py
import os
import torch
from types import SimpleNamespace

from auto_avsr.datamodule.data_module import DataModule
from auto_avsr.datamodule.transforms import TextTransform

from auto_avsr.espnet.nets.batch_beam_search import BatchBeamSearch
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from auto_avsr.espnet.nets.scorers.length_bonus import LengthBonus



def _get_beam_search_decoder(model, token_list, ctc_weight=0.1, lm_weight=0.0, penalty=0.0, beam_size=40):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
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


def load_model(ckpt_path: str, modality: str = "video", ctc_weight: float = 0.1):
    """
    Loads the exact E2E model used by eval.py, on CPU.
    Returns an engine dict used by run().
    """
    # CPU-friendly threading (mirrors eval.py intent)
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

    return {"model": model, "beam": beam, "text": text_transform}


@torch.no_grad()
def run(engine, model_input_tensor: torch.Tensor) -> str:
    """
    Runs the exact forward path from eval.py test_step, but without Lightning.
    model_input_tensor should match sample["input"] produced by DataModule.
    """
    model = engine["model"]
    beam = engine["beam"]
    text_transform = engine["text"]

    # Frontend → proj → encoder (same as eval.py)
    x = model.frontend(model_input_tensor.unsqueeze(0))
    x = model.proj_encoder(x)
    enc_feat, _ = model.encoder(x, None)
    enc_feat = enc_feat.squeeze(0)

    nbest = beam(enc_feat)
    # nbest entries are Hypothesis objects; yseq includes sos at index 0
    yseq = nbest[0].yseq[1:]
    token_ids = torch.tensor(list(map(int, yseq)))
    pred_text = text_transform.post_process(token_ids).replace("<eos>", "").strip()
    return pred_text


def run_from_roi_root(roi_root: str, test_file: str, ckpt_path: str, ctc_weight: float = 0.1) -> str:
    """
    Convenience wrapper: uses your existing DataModule to load the ROI clip
    described by labels/<test_file>, then runs CPU inference in-process.
    """
    args = SimpleNamespace(
        modality="video",
        root_dir=str(roi_root),
        test_file=str(test_file),
        pretrained_model_path=str(ckpt_path),
        ctc_weight=float(ctc_weight),
        # keep any other defaults the DataModule expects if it reads attributes
        decode_snr_target=999999,
        debug=False,
    )

    dm = DataModule(args)
    dm.setup(stage="test")
    dl = dm.test_dataloader()

    # Grab the first sample/batch from the test loader
    batch = next(iter(dl))

    # Many setups return a dict of tensors directly (single sample),
    # or a dict of batched tensors. We handle both.
    if isinstance(batch, dict) and "input" in batch:
        inp = batch["input"]
        # If batched, take first element
        if isinstance(inp, torch.Tensor) and inp.dim() >= 1 and inp.shape[0] > 1:
            # Heuristic: if loader batches, it'll be [B, ...]
            # but some pipelines use [T, H, W, C] per sample.
            # If it looks batched, take index 0.
            if inp.dim() >= 4:
                try:
                    inp0 = inp[0]
                    # If that drops batch correctly, use it.
                    inp = inp0
                except Exception:
                    pass
    else:
        raise RuntimeError(f"Unexpected batch format from DataModule: {type(batch)} keys={getattr(batch,'keys',lambda:[])()}")

    engine = load_model(ckpt_path, modality="video", ctc_weight=ctc_weight)
    return run(engine, inp)
