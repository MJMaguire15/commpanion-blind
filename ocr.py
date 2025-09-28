import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
import os
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancementLevel(Enum):
    """Image enhancement levels for preprocessing."""
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"
    CUSTOM = "custom"


@dataclass
class OCRResult:
    """Data class to store OCR detection results."""
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    level: int


@dataclass
class ImageInfo:
    """Data class to store image information."""
    path: str
    size: Tuple[int, int]
    mode: str
    file_size_kb: float


@dataclass
class AnalysisStats:
    """Data class to store analysis statistics."""
    total_characters: int
    total_words: int
    detected_elements: int
    average_confidence: float


class DocTROCR:
    """
    Professional DocTR OCR wrapper with advanced preprocessing and analysis capabilities.
    
    This class provides a comprehensive interface for optical character recognition
    using DocTR with optimized preprocessing and detailed analysis features.
    """
    
    def __init__(self, model_name: str = "db_resnet50") -> None:
        """
        Initialize DocTR OCR with model loading.
        
        Args:
            model_name: DocTR model name (default: db_resnet50)
        """
        self._image_cache: Dict[str, Any] = {}
        self.model_name = model_name
        
        # Initialize DocTR model
        logger.info("⏳ Initializing DocTR model... This might take a moment (first run only).")
        try:
            self.model = ocr_predictor(pretrained=True)
            logger.info("✅ DocTR model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DocTR model: {e}")
            raise
        
        # Configuration paths
        self.OCR_OUTPUT_DIR = "output"
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray]) -> Union[str, np.ndarray]:
        """
        Load and prepare image input for DocTR processing.
        
        Args:
            image_input: Image file path or numpy array
            
        Returns:
            Image path or numpy array ready for DocTR
        """
        if isinstance(image_input, (str, Path)):
            return str(image_input)
        elif isinstance(image_input, np.ndarray):
            # Save numpy array to temporary file for DocTR processing
            temp_path = "temp_doctr_image.jpg"
            cv2.imwrite(temp_path, image_input)
            return temp_path
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def preprocess_image(self, 
                        image_input: Union[str, Path, np.ndarray],
                        enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM,
                        min_size: int = 300) -> Union[str, np.ndarray]:
        """
        Apply preprocessing to optimize image for OCR (DocTR handles most preprocessing internally).
        
        Args:
            image_input: Input image in various formats
            enhancement_level: Level of preprocessing enhancement
            min_size: Minimum dimension for image resizing
            
        Returns:
            Preprocessed image path or array
        """
        # DocTR handles most preprocessing internally, so we do minimal processing
        if isinstance(image_input, (str, Path)):
            return str(image_input)
        elif isinstance(image_input, np.ndarray):
            # Basic preprocessing for numpy arrays
            image = image_input.copy()
            
            # Resize if too small
            height, width = image.shape[:2]
            if min(height, width) < min_size:
                ratio = min_size / min(height, width)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                logger.debug(f"Image resized from {width}x{height} to {new_width}x{new_height}")
            
            return image
        
        return image_input
    
    def extract_text(self, 
                    image_input: Union[str, Path, np.ndarray],
                    languages: Union[str, List[str]] = 'eng',  # Keep for compatibility
                    enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM,
                    custom_config: Optional[str] = None) -> str:
        """
        Extract text from image using DocTR OCR.
        
        Args:
            image_input: Input image in various formats
            languages: Language code(s) - kept for compatibility but DocTR is multilingual by default
            enhancement_level: Image preprocessing level
            custom_config: Custom configuration (unused but kept for compatibility)
            
        Returns:
            Extracted text as string
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_input, enhancement_level)
            
            # Handle numpy array input
            if isinstance(processed_image, np.ndarray):
                temp_path = "temp_doctr_extract.jpg"
                cv2.imwrite(temp_path, processed_image)
                image_path = temp_path
            else:
                image_path = processed_image
            
            start_time = time.perf_counter()
            
            # Load document and run OCR
            doc = DocumentFile.from_images(image_path)
            result = self.model(doc)
            
            # Extract text from result
            text_parts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if hasattr(word, 'value'):
                                text_parts.append(word.value)
            
            extracted_text = " ".join(text_parts)
            processing_time = time.perf_counter() - start_time
            
            logger.debug(f"Text extraction completed in {processing_time:.3f}s")
            
            # Clean up temporary file if created
            if isinstance(processed_image, np.ndarray) and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def extract_text_with_boxes(self, 
                               image_input: Union[str, Path, np.ndarray],
                               languages: Union[str, List[str]] = 'eng',
                               confidence_threshold: float = 0.3,
                               enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM) -> List[OCRResult]:
        """
        Extract text with bounding box coordinates and confidence scores.
        
        Args:
            image_input: Input image in various formats
            languages: Language code(s) - kept for compatibility
            confidence_threshold: Minimum confidence score (0-1 for DocTR)
            enhancement_level: Image preprocessing level
            
        Returns:
            List of OCRResult objects containing text, coordinates, and confidence
        """
        try:
            processed_image = self.preprocess_image(image_input, enhancement_level)
            
            # Handle numpy array input
            if isinstance(processed_image, np.ndarray):
                temp_path = "temp_doctr_boxes.jpg"
                cv2.imwrite(temp_path, processed_image)
                image_path = temp_path
                # Get image dimensions
                height, width = processed_image.shape[:2]
            else:
                image_path = processed_image
                # Load image to get dimensions
                img = cv2.imread(image_path)
                height, width = img.shape[:2]
            
            # Load document and run OCR
            doc = DocumentFile.from_images(image_path)
            result = self.model(doc)
            
            results = []
            
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if hasattr(word, 'value') and hasattr(word, 'confidence'):
                                confidence = float(word.confidence) * 100  # Convert to percentage
                                text = word.value.strip()
                                
                                # Filter by confidence and non-empty text
                                if confidence >= (confidence_threshold * 100) and text:
                                    # Convert relative coordinates to absolute pixels
                                    geometry = word.geometry
                                    x1, y1 = geometry[0]
                                    x2, y2 = geometry[1]
                                    
                                    abs_x1 = int(x1 * width)
                                    abs_y1 = int(y1 * height)
                                    abs_x2 = int(x2 * width)
                                    abs_y2 = int(y2 * height)
                                    
                                    result_obj = OCRResult(
                                        text=text,
                                        confidence=confidence,
                                        x=abs_x1,
                                        y=abs_y1,
                                        width=abs_x2 - abs_x1,
                                        height=abs_y2 - abs_y1,
                                        level=4  # Default level for word-level detection
                                    )
                                    results.append(result_obj)
            
            logger.info(f"Detected {len(results)} elements with confidence >= {confidence_threshold * 100}%")
            
            # Clean up temporary file if created
            if isinstance(processed_image, np.ndarray) and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Box extraction failed: {e}")
            return []
    
    def draw_detection_boxes(self, 
                           image_path: Union[str, Path],
                           results: List[OCRResult],
                           output_path: Union[str, Path] = "ocr_results.png",
                           show_confidence: bool = True,
                           min_confidence: float = 50.0,
                           box_colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> bool:
        """
        Draw OCR detection results on image with colored bounding boxes.
        
        Args:
            image_path: Path to original image
            results: List of OCRResult objects
            output_path: Output image path
            show_confidence: Include confidence scores in labels
            min_confidence: Minimum confidence for display
            box_colors: Custom colors for different confidence levels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Default color scheme based on confidence
            if box_colors is None:
                box_colors = {
                    'high': (0, 255, 0),    # Green for >80%
                    'medium': (0, 255, 255), # Yellow for 60-80%
                    'low': (0, 165, 255)     # Orange for <60%
                }
            
            drawn_count = 0
            for result in results:
                if result.confidence < min_confidence:
                    continue
                
                # Determine color based on confidence
                if result.confidence >= 80:
                    color = box_colors['high']
                elif result.confidence >= 60:
                    color = box_colors['medium']
                else:
                    color = box_colors['low']
                
                # Draw bounding box
                cv2.rectangle(image, 
                            (result.x, result.y), 
                            (result.x + result.width, result.y + result.height), 
                            color, 2)
                
                # Prepare label
                if show_confidence:
                    label = f"{result.text} ({result.confidence:.1f}%)"
                else:
                    label = result.text
                
                # Truncate long labels
                if len(label) > 30:
                    label = label[:27] + "..."
                
                # Draw label background and text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image, 
                            (result.x, result.y - 25), 
                            (result.x + label_size[0], result.y), 
                            color, -1)
                cv2.putText(image, label, (result.x, result.y - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                drawn_count += 1
            
            cv2.imwrite(str(output_path), image)
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Displayed {drawn_count}/{len(results)} boxes (confidence >= {min_confidence}%)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to draw detection boxes: {e}")
            return False

    def DocTRRead(self, frame: np.ndarray) -> str:
        """
        Performs OCR on a given image frame using DocTR.
        This method maintains compatibility with your original DocTRRead function.

        Args:
            frame (numpy.ndarray): The image frame to perform OCR on.

        Returns:
            str: The extracted text, "No Text Detected", or error message.
        """
        if frame is None:
            logger.error("No frame provided for OCR.")
            return "Error: No image frame available for OCR."

        # Run OCR using DocTR
        try:
            doc = DocumentFile.from_images([frame])
            result = self.model(doc)
            output = result.export()
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return "Error: OCR processing failed."

        # Check if any text was detected
        if not result.pages or not any(block.lines for page in result.pages for block in page.blocks):
            return "No Text Detected"

        # Save JSON output (optional, for debugging)
        json_name = f"frame_res_{int(time.time())}.json"
        json_path = os.path.join(self.OCR_OUTPUT_DIR, json_name)
        os.makedirs(self.OCR_OUTPUT_DIR, exist_ok=True)
        
        try:
            with open(json_path, "w") as f:
                json.dump(output, f, indent=1)
            logger.debug(f"OCR results saved to {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON output: {e}")

        # Extract text from the result
        words = []
        for page in output.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    for word in line.get("words", []):
                        value = word.get("value")
                        if value:
                            words.append(value)

        # Join words into a single string
        final_result = " ".join(words)
        logger.info(f"Extracted text: {final_result}")
        return final_result

    def analyze_image(self, 
                     image_path: Union[str, Path],
                     save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive OCR analysis on an image using DocTR.
        
        Args:
            image_path: Path to image file
            save_visualizations: Save result visualizations
            
        Returns:
            Comprehensive analysis report dictionary
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return {}
        
        logger.info(f"Starting comprehensive analysis: {image_path.name}")
        
        # Get image information
        try:
            import PIL.Image
            image = PIL.Image.open(image_path)
            file_size_kb = image_path.stat().st_size / 1024
            
            image_info = ImageInfo(
                path=str(image_path),
                size=image.size,
                mode=image.mode,
                file_size_kb=file_size_kb
            )
            
            logger.info(f"Image: {image_info.size[0]}x{image_info.size[1]}px, "
                       f"{image_info.mode}, {file_size_kb:.1f}KB")
            
        except Exception as e:
            logger.error(f"Failed to read image info: {e}")
            return {}
        
        # Analysis results
        analysis_report = {
            'image_info': image_info,
            'results': {},
            'timestamp': time.time()
        }
        
        try:
            # Extract text and boxes using DocTR
            text = self.extract_text(image_path)
            boxes = self.extract_text_with_boxes(image_path)
            
            # Calculate statistics
            stats = AnalysisStats(
                total_characters=len(text),
                total_words=len(text.split()) if text else 0,
                detected_elements=len(boxes),
                average_confidence=np.mean([box.confidence for box in boxes]) if boxes else 0.0
            )
            
            analysis_report['results']['doctr'] = {
                'text': text,
                'boxes': [box.__dict__ for box in boxes],
                'stats': stats.__dict__
            }
            
            logger.info(f"DocTR analysis: {stats.total_words} words, "
                       f"{stats.detected_elements} elements, "
                       f"avg confidence: {stats.average_confidence:.1f}%")
            
            # Save visualization if requested
            if save_visualizations and boxes:
                output_path = f"analysis_doctr_{image_path.stem}.png"
                self.draw_detection_boxes(image_path, boxes, output_path)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_report['results']['doctr'] = {'error': str(e)}
        
        logger.info("Analysis completed successfully")
        return analysis_report
    
    def clear_cache(self) -> None:
        """Clear internal image cache to free memory."""
        self._image_cache.clear()
        logger.debug("Image cache cleared")

    def extract_text_from_frame(self,
                            frame: np.ndarray,
                            enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM) -> Optional[str]:
        """
        Perform OCR on an already captured image (np.ndarray).

        Args:
            frame: Captured image
            enhancement_level: Image enhancement level before OCR

        Returns:
            Extracted text or None if failed
        """
        try:
            text = self.extract_text(frame, enhancement_level=enhancement_level)
            return text
        except Exception as e:
            logger.error(f"Error extracting OCR from frame: {e}")
            return None