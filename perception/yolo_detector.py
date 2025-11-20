"""
YOLO Detector Module

Handles object detection using YOLOv8.
Provides clean interface for detecting objects in images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional
from config import DetectionConfig


class Detection:
    """
    Data class representing a single detection
    """
    def __init__(self, class_name: str, confidence: float, bbox: tuple):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x_min, y_min, x_max, y_max)
        
        # Calculate center of bounding box
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        
        # Calculate size
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
    
    def __repr__(self):
        return (f"Detection(class={self.class_name}, "
                f"conf={self.confidence:.2f}, "
                f"center=({self.center_x:.0f}, {self.center_y:.0f}))")


class YOLODetector:
    """
    Wrapper for YOLO model with detection logic
    """
    
    def __init__(self, model_path: str = None, target_class: str = None, 
                 confidence_threshold: float = None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model (uses config default if None)
            target_class: Class to detect (uses config default if None)
            confidence_threshold: Minimum confidence (uses config default if None)
        """
        # Use config defaults if not specified
        self.model_path = model_path or DetectionConfig.MODEL_PATH
        self.target_class = target_class or DetectionConfig.TARGET_CLASS
        self.confidence_threshold = confidence_threshold or DetectionConfig.CONFIDENCE_THRESHOLD
        
        # Load YOLO model
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        print(f"YOLO model loaded. Target class: {self.target_class}")
        
        # Detection history for confirmation
        self.detection_history = []
        self.required_detections = DetectionConfig.REQUIRED_DETECTIONS
        
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on an image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Run YOLO inference
        results = self.model(image, verbose=False)
        
        detections = []
        
        # Parse results
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Extract detection info
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                # Filter by target class and confidence
                if class_name == self.target_class and confidence >= self.confidence_threshold:
                    # Extract bounding box
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = tuple(map(int, xyxy))  # (x_min, y_min, x_max, y_max)
                    
                    # Create Detection object
                    detection = Detection(class_name, confidence, bbox)
                    detections.append(detection)
        
        return detections
    
    def detect_target(self, image: np.ndarray) -> Optional[Detection]:
        """
        Detect target class in image (returns best detection)
        
        Args:
            image: OpenCV image
            
        Returns:
            Best Detection object or None if no target found
        """
        detections = self.detect(image)
        
        if len(detections) == 0:
            return None
        
        # Return detection with highest confidence
        best_detection = max(detections, key=lambda d: d.confidence)
        return best_detection
    
    def update_detection_history(self, detected: bool) -> bool:
        """
        Track detection over time to confirm target
        
        Args:
            detected: True if target was detected in current frame
            
        Returns:
            True if target is confirmed (enough consecutive detections)
        """
        # Add to history
        self.detection_history.append(detected)
        
        # Keep only recent history
        if len(self.detection_history) > self.required_detections:
            self.detection_history.pop(0)
        
        # Check if we have enough consecutive detections
        if len(self.detection_history) >= self.required_detections:
            return all(self.detection_history)
        
        return False
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image: OpenCV image
            detections: List of Detection objects
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for det in detections:
            # Draw bounding box
            cv2.rectangle(
                result_image,
                (det.bbox[0], det.bbox[1]),
                (det.bbox[2], det.bbox[3]),
                (0, 255, 0),  # Green
                2
            )
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(
                result_image,
                label,
                (det.bbox[0], det.bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw center point
            cv2.circle(
                result_image,
                (int(det.center_x), int(det.center_y)),
                5,
                (0, 0, 255),  # Red
                -1
            )
        
        return result_image


# Test function
if __name__ == "__main__":
    """
    Test YOLO detector with a sample image
    """
    detector = YOLODetector()
    
    # Load test image
    test_image = cv2.imread('/workspace/test_image.jpg')
    
    if test_image is not None:
        # Run detection
        detections = detector.detect(test_image)
        
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  {det}")
        
        # Draw and save result
        result = detector.draw_detections(test_image, detections)
        cv2.imwrite('/workspace/detection_result.jpg', result)
        print("Saved result to /workspace/detection_result.jpg")
    else:
        print("No test image found at /workspace/test_image.jpg")
