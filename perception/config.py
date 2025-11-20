"""
Configuration file for UAV Emergency Response System

Contains all system parameters in one place for easy tuning.
"""

class DroneConfig:
    """Drone connection and flight parameters"""
    
    # Connection
    MAVLINK_ADDRESS = "udp://:14540"
    
    # Flight parameters
    TAKEOFF_ALTITUDE = 5.0  # meters
    CRUISE_SPEED = 2.0      # m/s
    POSITION_THRESHOLD = 0.5  # meters (how close = "reached")
    
    # Safety
    MAX_ALTITUDE = 10.0     # meters
    MIN_ALTITUDE = 2.0      # meters
    RTL_ALTITUDE = 5.0      # Return to launch altitude


class DetectionConfig:
    """YOLO detection parameters"""
    
    # Model
    MODEL_PATH = "/workspace/yolov8n.pt"
    TARGET_CLASS = "person"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Detection confirmation
    REQUIRED_DETECTIONS = 3  # Consecutive detections to confirm


class CameraConfig:
    """Camera parameters"""
    
    # From Gazebo camera specs
    WIDTH = 1280
    HEIGHT = 960
    FOCAL_LENGTH_X = 539.9
    FOCAL_LENGTH_Y = 539.9
    
    # ROS topic
    CAMERA_TOPIC = "/camera/image_raw"


class PathConfig:
    """Path planning parameters"""
    
    # Search pattern type: "grid", "spiral", "waypoints"
    PATTERN_TYPE = "grid"
    
    # Grid search parameters
    GRID_SIZE = 20  # meters
    GRID_SPACING = 5  # meters between lines
    
    # Waypoints for manual pattern
    SEARCH_WAYPOINTS = [
        (5.0, 0.0, -5.0, "North 5m"),
        (10.0, 0.0, -5.0, "North 10m"),
        (10.0, 5.0, -5.0, "Northeast corner"),
        (5.0, 5.0, -5.0, "East 5m"),
        (0.0, 5.0, -5.0, "Southeast corner"),
        (0.0, 0.0, -5.0, "Home position")
    ]
