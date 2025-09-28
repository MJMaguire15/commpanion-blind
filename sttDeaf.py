from qai_hub_models.models.whisper_base_en.model import WhisperBaseEn
from qai_hub_models.models._shared.whisper.app import WhisperApp
from qai_hub_models.utils.onnx_torch_wrapper import OnnxModelTorchWrapper
from pathlib import Path


class SpeechToTextApplication:
    """
    Application for transcribing speech from audio files using WhisperBaseEn.
    """

    def __init__(self) -> None:
        """
        Initialize the SpeechToTextApplication.
        """
        self.model = WhisperBaseEn.from_pretrained()
        self.app = WhisperApp(
            OnnxModelTorchWrapper.OnNPU("models\whisper_base_en-whisperencoderinf.onnx"),
            OnnxModelTorchWrapper.OnNPU("models\whisper_base_en-whisperdecoderinf.onnx"),
            num_decoder_blocks=self.model.num_decoder_blocks,
            num_decoder_heads=self.model.num_decoder_heads,
            attention_dim=self.model.attention_dim,
            mean_decode_len=self.model.mean_decode_len,
        )

    def transcribe(self, audio_file_path: str | Path) -> str:
        """
        Transcribe a specific audio file.

        Args:
            audio_file_path (str | Path): The path to the audio file to transcribe.

        Returns:
            str: The transcription result.
        """
        # Make sure the rest of the function code is also updated
        # as per the previous answer.
        if not Path(audio_file_path).exists():
            return "" 

        transcription = self.app.transcribe(str(audio_file_path), audio_sample_rate=None)
        print(f"Transcription result: {transcription}")

        try:
            Path(audio_file_path).unlink()
            print(f"Deleted audio file: {audio_file_path}")
        except OSError as e:
            print(f"Error deleting file {audio_file_path}: {e}")

        return transcription