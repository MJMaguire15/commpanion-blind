# eval.py  — fixed
import logging
from argparse import ArgumentParser
import os
import torch
import torchaudio
import threading

from datamodule.data_module import DataModule
from datamodule.transforms import TextTransform

from cosine import WarmupCosineScheduler
from pytorch_lightning import Trainer, LightningModule
import pytorch_lightning as pl

from auto_avsr.espnet.nets.batch_beam_search import BatchBeamSearch
from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from auto_avsr.espnet.nets.scorers.length_bonus import LengthBonus


# ---- runtime setup (CPU friendly) ----
torchaudio.set_audio_backend("soundfile")  # ok even if deprecated; harmless no-op
torch.set_num_threads(min(8, os.cpu_count() or 8))
torch.set_num_interop_threads(1)
logging.basicConfig(level=logging.WARNING)


# ---------- helpers ----------
def compute_word_level_distance(seq1: str, seq2: str) -> int:
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0.0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
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


# ---------- Lightning module ----------
class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(
            len(self.token_list),
            self.modality,
            ctc_weight=getattr(args, "ctc_weight", 0.1),
        )

        # --- load checkpoint (full model by default)
        if getattr(args, "pretrained_model_path", None):
            ckpt = torch.load(args.pretrained_model_path, map_location="cpu")
            if getattr(args, "transfer_frontend", False):
                tmp = {
                    k: v
                    for k, v in ckpt["model_state_dict"].items()
                    if k.startswith("trunk.") or k.startswith("frontend3D.")
                }
                self.model.frontend.load_state_dict(tmp, strict=False)
                print("Pretrained weights of the frontend component are loaded successfully.")
            elif getattr(args, "transfer_encoder", False):
                tmp = {k.replace("frontend.", ""): v for k, v in ckpt.items() if k.startswith("frontend.")}
                self.model.frontend.load_state_dict(tmp, strict=False)
                tmp = {k.replace("proj_encoder.", ""): v for k, v in ckpt.items() if k.startswith("proj_encoder.")}
                self.model.proj_encoder.load_state_dict(tmp, strict=False)
                tmp = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp, strict=False)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
            else:
                # ckpt may be a plain state_dict in your setup
                state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                self.model.load_state_dict(state, strict=False)
                print("Pretrained weights of the full model are loaded successfully.")

        # built at test time; defined here for type hints
        self.beam_search = None
        self.total_length = 0
        self.total_edit_distance = 0
        self.total_char_length = 0
        self.total_char_edit_distance = 0

    # ------- training API (unused for eval, kept intact) -------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.98),
        )
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) / max(
            1, self.trainer.num_devices * self.trainer.num_nodes
        )
        scheduler = WarmupCosineScheduler(
            optimizer, self.args.warmup_epochs, self.args.max_epochs, steps_per_epoch
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # ------- evaluation -------
    @torch.no_grad()
    def test_step(self, sample, batch_idx):
        """
        Expecting sample with keys like:
          - 'input': (T, H, W, C) already converted to the model's expected tensor in the datamodule
          - optional 'target': token ids (may be absent or dummy for unlabeled inference)
        """
        # Build decoder on first use
        if self.beam_search is None:
            self.beam_search = get_beam_search_decoder(
                self.model, self.token_list, ctc_weight=getattr(self.args, "ctc_weight", 0.1)
            )

        # Frontend → proj → encoder
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)

        # Beam search (top-1)
        nbest = self.beam_search(enc_feat)
        nbest = [h.asdict() for h in nbest[:1]]
        token_ids = torch.tensor(list(map(int, nbest[0]["yseq"][1:])))  # drop sos
        pred_text = self.text_transform.post_process(token_ids).replace("<eos>", "").strip()

        # If a real target exists, update WER stats; otherwise skip
        if "target" in sample and isinstance(sample["target"], torch.Tensor) and sample["target"].numel() > 0:
            tgt_text = self.text_transform.post_process(sample["target"]).strip()
            if tgt_text:
                self.total_edit_distance += compute_word_level_distance(tgt_text, pred_text)
                self.total_length += len(tgt_text.split())

        # *** CRUCIAL: return prediction so callbacks can print it ***
        return {"pred_text": pred_text}

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.total_char_length = 0
        self.total_char_edit_distance = 0

    def on_test_epoch_end(self):
        # Guard metrics when no references are present
        tl = getattr(self, "total_length", 0)
        self.log("wer", (self.total_edit_distance / tl) if tl > 0 else 0.0)

        if hasattr(self, "total_char_length"):
            tcl = getattr(self, "total_char_length", 0)
            cer = (self.total_char_edit_distance / tcl) if tcl and tcl > 0 else 0.0
            self.log("cer", cer)


# ---------- print transcript callback ----------
class PrintPredCallback(pl.Callback):
    def __init__(self, speak: bool = False):
        super().__init__()
        self.speak = speak
        self._engine = None
        self._lock = threading.Lock()
        self._last_spoken = ""

        if self.speak:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 180)
                self._engine.setProperty("volume", 1.0)
            except Exception as e:
                print("[TTS] init failed:", e)
                self.speak = False

    def _say(self, text: str):
        if not self._engine: return
        # de-duplicate repeated outputs
        with self._lock:
            if text.strip() and text.strip() != self._last_spoken:
                self._last_spoken = text.strip()
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print("[TTS] speak failed:", e)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        try:
            spoken = None
            if isinstance(outputs, dict):
                for k in ("pred_text", "pred", "hyp", "hyp_text", "text", "transcript"):
                    if k in outputs and outputs[k]:
                        val = outputs[k]
                        if isinstance(val, (list, tuple)):
                            for s in val: print(f"[DECODE] {s}")
                            spoken = " ".join(map(str, val))
                        else:
                            print(f"[DECODE] {val}")
                            spoken = str(val)
                        break
            else:
                print("[DECODE] outputs type:", type(outputs))

            if self.speak and spoken:
                self._say(spoken)
        except Exception as e:
            print("[DECODE] (could not print/speak prediction)", e)



# ---------- trainer / arg parsing ----------
def get_trainer(args):
    return Trainer(
        num_nodes=1,
        devices=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=True,
        callbacks=[PrintPredCallback(speak=getattr(args, "speak", False))],
    )

def get_lightning_module(args):
    # Use the ModelModule defined in THIS file
    return ModelModule(args)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality", type=str, required=True, choices=["audio", "video"],
        help="Type of input modality",
    )
    parser.add_argument(
        "--root-dir", type=str, required=True,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--test-file", type=str, required=True,
        help="Filename of testing label list, e.g. list.csv",
    )
    parser.add_argument(
        "--pretrained-model-path", type=str, required=True,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--decode-snr-target", type=float, default=999999,
        help="Level of SNR (unused for pure video, but kept for compatibility)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    # Training-only args (kept for completeness)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--ctc_weight", type=float, default=0.1)
    parser.add_argument("--speak", action="store_true", help="Speak decoded text with TTS")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args)
    trainer = get_trainer(args)
    # Running test will now print the transcript via callback
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
