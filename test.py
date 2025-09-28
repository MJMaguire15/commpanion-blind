#!/usr/bin/env python3
"""
Example script demonstrating how to use CameraManager with BlipModel
for automated image capture and captioning.
"""

import os
import time
import logging
from camera_manager import CameraManager
from blip import BlipModel  # Assuming your BlipModel is in blip_model.py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    
    # Initialize camera manager
    camera_manager = CameraManager(
        camera_id=2,  # Use default camera
        image_dir="image"
    )
    
    # Initialize BLIP model
    blip_model = BlipModel()
    
    try:
        # Test camera availability
        logger.info("Testing camera availability...")
        camera_info = camera_manager.get_camera_info()
        if not camera_info.get("available", False):
            logger.error("Camera not available!")
            return
        
        logger.info(f"Camera info: {camera_info}")
        
        # Load BLIP model
        logger.info("Loading BLIP model...")
        blip_model.load()
        
        # Example 1: Basic usage as requested
        logger.info("\n=== Example 1: Basic Usage ===")
        image_path = camera_manager.take_picture()
        if image_path:
            caption = blip_model.generate_caption(image_path)
            logger.info(f"Image: {image_path}")
            logger.info(f"Caption: {caption}")
        else:
            logger.error("Failed to capture image")
        
        # Example 2: Multiple captures with different parameters
        logger.info("\n=== Example 2: Multiple Captures ===")
        for i in range(3):
            logger.info(f"Capture {i+1}/3...")
            image_path = camera_manager.take_picture()
            if image_path:
                # Generate caption with custom parameters
                caption = blip_model.generate_caption(
                    image_path,
                    max_length=30,
                    num_beams=3,
                    temperature=0.7
                )
                logger.info(f"Image {i+1}: {os.path.basename(image_path)}")
                logger.info(f"Caption {i+1}: {caption}")
            
            # Wait between captures
            if i < 2:
                time.sleep(2)
        
        # Example 3: Batch capture and caption
        logger.info("\n=== Example 3: Batch Processing ===")
        batch_images = camera_manager.take_multiple_pictures(count=2, interval=3.0)
        
        for idx, image_path in enumerate(batch_images, 1):
            caption = blip_model.generate_caption(image_path)
            logger.info(f"Batch {idx}: {os.path.basename(image_path)} -> {caption}")
        
        # Example 4: Using BlipModel's built-in capture (for comparison)
        logger.info("\n=== Example 4: BlipModel Built-in Capture ===")
        blip_image_path, blip_caption = blip_model.capture_and_describe()
        if blip_image_path and blip_caption:
            logger.info(f"BLIP capture: {os.path.basename(blip_image_path)} -> {blip_caption}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        blip_model.unload()
        logger.info("Done!")


def interactive_mode():
    """Interactive mode for testing"""
    camera_manager = CameraManager(camera_id=0)
    blip_model = BlipModel()
    
    try:
        blip_model.load()
        
        while True:
            input("\nPress Enter to take a picture and generate caption (or Ctrl+C to quit)...")
            
            # Take picture
            image_path = camera_manager.take_picture()
            if not image_path:
                print("Failed to capture image!")
                continue
            
            # Generate caption
            caption = blip_model.generate_caption(image_path)
            
            print(f"\n📸 Image saved: {image_path}")
            print(f"🏷️  Caption: {caption}")
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    finally:
        blip_model.unload()


def quick_capture_and_caption():
    """Quick function for single capture and caption"""
    camera_manager = CameraManager()
    blip_model = BlipModel()
    
    try:
        blip_model.load()
        
        # As requested in the prompt
        image = camera_manager.take_picture()
        caption = blip_model.generate_caption(image)
        
        return image, caption
        
    finally:
        blip_model.unload()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        image_path, caption = quick_capture_and_caption()
        print(f"Image: {image_path}")
        print(f"Caption: {caption}")
    else:
        main()