# inference_cpu_stream.py
import torch
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from auto_avsr.datamodule.transforms import TextTransform

from auto_avsr.espnet.nets.batch_beam_search import BatchBeamSearch
from auto_avsr.espnet.nets.scorers.length_bonus import LengthBonus

torch.set_grad_enabled(False)

# ===================== DECODE MODE =====================
USE_BEAM = True        # True = beam (accuracy), False = greedy (speed)
BEAM_SIZE = 3          # Only used if USE_BEAM = True
# ======================================================


def build_beam(model, token_list, beam_size):
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = {
        "decoder": 0.7,
        "ctc": 0.3,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.odim - 1,
        eos=model.odim - 1,
        token_list=token_list,
        pre_beam_score_key="decoder",
    )


def load_model(ckpt_path: str):
    text = TextTransform()
    token_list = text.token_list

    model = E2E(
        len(token_list),
        modality="video",
        ctc_weight=0.1,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    engine = {
        "model": model,
        "text": text,
    }

    if USE_BEAM:
        engine["beam"] = build_beam(model, token_list, BEAM_SIZE)

    return engine


@torch.no_grad()
def run(engine, clip: torch.Tensor) -> str:
    """
    clip: (T, 1, H, W), float32 in [0,1]
    """
    model = engine["model"]
    text = engine["text"]

    if clip.dim() != 4 or clip.shape[1] != 1:
        raise RuntimeError(f"Bad clip shape: {clip.shape}")

    # ESPNet expects (B, T, C, H, W)
    x = clip.unsqueeze(0)  # (1, T, 1, H, W)

    # ---- Forward ----
    x = model.frontend(x)
    x = model.proj_encoder(x)
    enc, _ = model.encoder(x, None)
    enc = enc.squeeze(0)

    # ---- Decode ----
    if USE_BEAM:
        beam = engine["beam"]
        nbest = beam(enc)
        yseq = nbest[0].yseq[1:]  # drop <sos>
        tokens = torch.tensor(list(map(int, yseq)))
    else:
        logits = model.ctc.ctc_lo(enc)
        pred = logits.argmax(dim=-1)
        tokens = torch.unique_consecutive(pred)
        tokens = tokens[tokens != model.blank]

    return text.post_process(tokens).replace("<eos>", "").strip()
