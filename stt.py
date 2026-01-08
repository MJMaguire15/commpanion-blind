from pathlib import Path

from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn
from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper


class SpeechToTextApplication:
    """
    Application for transcribing speech from audio files using WhisperBaseEn.
    """

    def __init__(self, audio_records_path: Path | str | None = None) -> None:
        # Load Whisper Base English model metadata
        self.model = WhisperBaseEn.from_pretrained()

        # Create Whisper inference app using existing ONNX files
        self.app = WhisperApp(
            OnnxModelTorchWrapper.OnCPU(
                r"models\whisper_base_en-whisperencoderinf.onnx"
            ),
            OnnxModelTorchWrapper.OnCPU(
                r"models\whisper_base_en-whisperdecoderinf.onnx"
            ),
            num_decoder_blocks=self.model.num_decoder_blocks,
            num_decoder_heads=self.model.num_decoder_heads,
            attention_dim=self.model.attention_dim,
            mean_decode_len=self.model.mean_decode_len,
        )

        # Audio directory
        if isinstance(audio_records_path, str):
            self.audio_records_path: Path | None = Path(audio_records_path)
        else:
            self.audio_records_path: Path | None = audio_records_path

        self.last_audio_file: Path | None = None

    def _get_audio_file(self) -> Path:
        if self.audio_records_path is None:
            raise ValueError("Audio records path is not set.")

        audio_files = sorted(
            self.audio_records_path.glob("*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not audio_files:
            raise FileNotFoundError("No audio files found.")

        self.last_audio_file = audio_files[0]
        return audio_files[0]

    def _delete_audio_file(self) -> None:
        if self.last_audio_file and self.last_audio_file.exists():
            self.last_audio_file.unlink()
            print(f"Deleted audio file: {self.last_audio_file}")
            self.last_audio_file = None

    def transcribe(self) -> str:
        audio_file = self._get_audio_file()
        transcription = self.app.transcribe(
            str(audio_file), audio_sample_rate=None
        )
        print(f"Transcription result: {transcription}")
        self._delete_audio_file()
        return transcription
