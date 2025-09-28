import os
import torch
import threading
import logging
import numpy as np
from PIL import Image
from typing import Union
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipModel:
    """Thread-safe BLIP model for image captioning"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        image_dir: str = "image"
    ):
        self.model_name = model_name
        self.image_dir = image_dir

        os.makedirs(self.image_dir, exist_ok=True)

        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_lock = threading.Lock()
        self._loaded = False

        self.logger = logging.getLogger("BlipModel")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load(self):
        """Load the BLIP model"""
        self.logger.info("🔄 Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self._loaded = True
        self.logger.info("✅ BLIP model loaded.")

    def generate_caption(self, image: Union[str, Image.Image, np.ndarray],
                        max_length: int = 50,
                        num_beams: int = 5,
                        early_stopping: bool = True,
                        temperature: float = 1.0) -> str:
        """Generate a caption from an image."""
        caption = "Caption generation failed."

        try:
            # Convert to PIL image directly
            pil_image = self._convert_to_pil(image)

            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                caption_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    temperature=temperature,
                    do_sample=False
                )

            caption = self.processor.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
            self.logger.info(f"Generated caption: {caption}")

        except Exception as e:
            self.logger.error(f"Error during caption generation: {e}")

        return caption

    def describe_frame(self, frame: np.ndarray, caption_params: dict = None) -> str:
        """
        Use an already captured image (numpy frame) to generate a description.

        Args:
            frame: Captured image (numpy.ndarray)
            caption_params: Optional parameters for caption generation

        Returns:
            str: Generated description
        """
        if caption_params is None:
            caption_params = {}

        try:
            caption = self.generate_caption(frame, **caption_params)
            return caption
        except Exception as e:
            self.logger.error(f"Error during caption generation from frame: {e}")
            return "Caption generation failed."

    def _convert_to_pil(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert any supported image format to a PIL image."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR format from OpenCV and convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                image_rgb = image
            return Image.fromarray(image_rgb)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def unload(self):
        """Unload model from memory."""
        with self._model_lock:
            if self._loaded:
                self.model = None
                self.processor = None
                self._loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Model unloaded successfully")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass