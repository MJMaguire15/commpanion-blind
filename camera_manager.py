import os
import cv2
import time
import logging
import numpy as np
from contextlib import contextmanager
from typing import Optional, Union


class CameraManager:
    """Thread-safe camera manager for image capture"""
    
    def __init__(
        self,
        camera_id: int = 0,
        image_dir: str = "captured",
        image_width: int = 640,
        image_height: int = 480,
        fps: int = 30
    ):
        """
        Initialize Camera Manager
        
        Args:
            camera_id: Camera index (0 for default camera)
            image_dir: Directory to save captured images
            image_width: Camera frame width
            image_height: Camera frame height
            fps: Camera FPS setting
        """
        self.camera_id = camera_id
        self.image_dir = image_dir
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        
        # Create image directory if it doesn't exist
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("CameraManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def _get_camera(self, camera_id: Optional[int] = None):
        """Safe camera access with context manager"""
        cam_id = camera_id if camera_id is not None else self.camera_id
        cap = cv2.VideoCapture(cam_id)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot access camera {cam_id}")
            
            # Configure camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def take_picture(
        self,
        output_path: Optional[str] = None,
        camera_id: Optional[int] = None,
        stabilization_time: float = 0.1,
        warmup_frames: int = 1,
        quality_frames: int = 5
    ) -> Optional[str]:
        """
        Capture a single image from the camera
        
        Args:
            output_path: Path to save the image (auto-generated if None)
            camera_id: Camera index to use (uses instance default if None)
            stabilization_time: Time to wait for camera stabilization
            warmup_frames: Number of frames to skip for camera warmup
            quality_frames: Number of frames to capture for quality selection
        
        Returns:
            Path to the captured image or None if failed
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = int(time.time() * 1000)
            output_path = os.path.join(self.image_dir, f"capture_{timestamp}.jpg")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cam_id = camera_id if camera_id is not None else self.camera_id
        
        try:
            with self._get_camera(cam_id) as cap:
                self.logger.info(f"Initializing camera {cam_id} for capture...")
                
                # Warmup frames
                for _ in range(warmup_frames):
                    ret, _ = cap.read()
                    if not ret:
                        self.logger.error("Failed to read frames during warmup")
                        return None
                    time.sleep(0.1)
                
                # Stabilization time
                time.sleep(stabilization_time)
                
                # Capture multiple frames and select the best one
                best_frame = None
                best_brightness = -1
                
                for i in range(quality_frames):
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(f"Failed to capture frame {i+1}/{quality_frames}")
                        continue
                    
                    # Calculate frame brightness as quality metric
                    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    
                    if brightness > best_brightness:
                        best_brightness = brightness
                        best_frame = frame.copy()
                    
                    time.sleep(0.1)
                
                if best_frame is None:
                    self.logger.error("Failed to capture any usable frame")
                    return None
                
                # Save the best frame
                success = cv2.imwrite(output_path, best_frame)
                if not success:
                    self.logger.error(f"Failed to save image to {output_path}")
                    return None
                
                self.logger.info(f"📸 Image captured: {output_path} (Brightness: {best_brightness:.1f})")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Error during image capture: {e}")
            return None
    
    def take_multiple_pictures(
        self,
        count: int = 3,
        interval: float = 1.0,
        output_dir: Optional[str] = None,
        camera_id: Optional[int] = None
    ) -> list[str]:
        """
        Capture multiple images with specified interval
        
        Args:
            count: Number of images to capture
            interval: Time interval between captures (seconds)
            output_dir: Directory to save images (uses instance default if None)
            camera_id: Camera index to use (uses instance default if None)
        
        Returns:
            List of paths to captured images
        """
        if output_dir is None:
            output_dir = self.image_dir
        
        captured_images = []
        timestamp_base = int(time.time() * 1000)
        
        for i in range(count):
            output_path = os.path.join(output_dir, f"capture_{timestamp_base}_{i+1:03d}.jpg")
            
            image_path = self.take_picture(output_path, camera_id)
            if image_path:
                captured_images.append(image_path)
                self.logger.info(f"Captured {i+1}/{count}: {image_path}")
            else:
                self.logger.warning(f"Failed to capture image {i+1}/{count}")
            
            # Wait before next capture (except for the last one)
            if i < count - 1:
                time.sleep(interval)
        
        return captured_images
    
    def test_camera(self, camera_id: Optional[int] = None, duration: float = 5.0) -> bool:
        """
        Test camera functionality by showing live preview
        
        Args:
            camera_id: Camera index to test (uses instance default if None)
            duration: Duration to show preview (seconds)
        
        Returns:
            True if camera works, False otherwise
        """
        cam_id = camera_id if camera_id is not None else self.camera_id
        
        try:
            with self._get_camera(cam_id) as cap:
                self.logger.info(f"Testing camera {cam_id} for {duration} seconds...")
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.error("Failed to read frame during test")
                        return False
                    
                    cv2.imshow(f"Camera {cam_id} Test", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.logger.info(f"Camera {cam_id} test completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Camera test failed: {e}")
            return False
        finally:
            cv2.destroyAllWindows()
    
    def cleanup_images(self, directory: Optional[str] = None, pattern: str = "capture_*.jpg"):
        """
        Clean up captured images
        
        Args:
            directory: Directory to clean (uses instance default if None)
            pattern: File pattern to match for deletion
        """
        import glob
        
        if directory is None:
            directory = self.image_dir
        
        pattern_path = os.path.join(directory, pattern)
        files_to_delete = glob.glob(pattern_path)
        
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
                self.logger.info(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                self.logger.error(f"Error deleting {file_path}: {e}")
        
        self.logger.info(f"Cleanup completed: {deleted_count} files deleted")
    
    def get_camera_info(self, camera_id: Optional[int] = None) -> dict:
        """
        Get camera information and capabilities
        
        Args:
            camera_id: Camera index to query (uses instance default if None)
        
        Returns:
            Dictionary with camera information
        """
        cam_id = camera_id if camera_id is not None else self.camera_id
        info = {"camera_id": cam_id, "available": False}
        
        try:
            with self._get_camera(cam_id) as cap:
                info.update({
                    "available": True,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                    "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
                    "saturation": cap.get(cv2.CAP_PROP_SATURATION),
                    "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                })
                
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            info["error"] = str(e)
        
        return info
    
    def __repr__(self):
        return f"CameraManager(camera_id={self.camera_id}, image_dir='{self.image_dir}')"