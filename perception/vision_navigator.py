"""
Vision-based Navigation Module
Converts YOLO detections to navigation waypoints using camera geometry

Uses camera intrinsics to transform pixel coordinates to world positions.

Author: Amr Hassan
Date: November 2024
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict


class CameraGeometry:
    """
    Handle camera coordinate transformations for the PX4 mono_cam sensor
    
    Camera frame convention:
    - X: Right
    - Y: Down
    - Z: Forward (optical axis)
    
    World frame (NED - North-East-Down):
    - X: North
    - Y: East
    - Z: Down
    """
    
    def __init__(self):
        """Initialize with PX4 mono_cam parameters from Gazebo"""
        
        # Image dimensions
        self.img_width = 1280
        self.img_height = 960
        
        # Camera intrinsics (from camera_info topic)
        self.fx = 539.936  # Focal length x (pixels)
        self.fy = 539.936  # Focal length y (pixels)
        self.cx = 640.0    # Principal point x (image center)
        self.cy = 480.0    # Principal point y (image center)
        
        # Camera mounting (from PX4 x500 model)
        # Camera is mounted facing forward and slightly down
        self.camera_tilt = -15.0  # degrees (negative = pointing down)
        self.camera_offset = np.array([0.12, 0.0, 0.0])  # meters [forward, right, down] from drone center
        
        # Field of view (calculated from focal length and image size)
        self.fov_horizontal = 2 * math.atan(self.img_width / (2 * self.fx))
        self.fov_vertical = 2 * math.atan(self.img_height / (2 * self.fy))
        
        print(f"Camera initialized:")
        print(f"  Resolution: {self.img_width}x{self.img_height}")
        print(f"  Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}")
        print(f"  FOV: H={math.degrees(self.fov_horizontal):.1f}°, V={math.degrees(self.fov_vertical):.1f}°")
    
    def pixel_to_camera_ray(self, pixel_x: int, pixel_y: int) -> np.ndarray:
        """
        Convert pixel coordinates to a ray in camera frame
        
        Args:
            pixel_x: X coordinate in image (0 = left)
            pixel_y: Y coordinate in image (0 = top)
        
        Returns:
            Unit vector [x, y, z] in camera frame
        """
        # Normalize pixel coordinates (center at principal point)
        x_normalized = (pixel_x - self.cx) / self.fx
        y_normalized = (pixel_y - self.cy) / self.fy
        
        # Camera frame: X=right, Y=down, Z=forward
        # Ray direction in camera frame
        ray = np.array([x_normalized, y_normalized, 1.0])
        
        # Normalize to unit vector
        ray = ray / np.linalg.norm(ray)
        
        return ray
    
    def camera_to_body_frame(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to drone body frame
        
        Args:
            point_camera: Point [x, y, z] in camera frame
        
        Returns:
            Point [x, y, z] in body frame (FRD: Forward-Right-Down)
        """
        # Rotation matrix for camera tilt (rotation around Y-axis)
        tilt_rad = math.radians(self.camera_tilt)
        cos_t = math.cos(tilt_rad)
        sin_t = math.sin(tilt_rad)
        
        # Rotation matrix: R_y(tilt)
        R = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0],
            [-sin_t, 0, cos_t]
        ])
        
        # Rotate point
        point_body = R @ point_camera
        
        # Add camera offset
        point_body += self.camera_offset
        
        return point_body
    
    def body_to_world_frame(self, 
                           point_body: np.ndarray, 
                           drone_position: Tuple[float, float, float],
                           drone_yaw: float) -> np.ndarray:
        """
        Transform point from body frame to world frame (NED)
        
        Args:
            point_body: Point [x, y, z] in body frame
            drone_position: Drone position [x, y, z] in world frame (NED)
            drone_yaw: Drone yaw angle in radians (0 = North)
        
        Returns:
            Point [x, y, z] in world frame (NED)
        """
        # Rotation matrix for yaw (rotation around Z-axis)
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        
        # Rotation matrix: R_z(yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        # Rotate point to world frame
        point_world = R @ point_body
        
        # Add drone position
        point_world += np.array(drone_position)
        
        return point_world
    
    def estimate_target_ground_position(self,
                                       bbox_center: Tuple[int, int],
                                       drone_position: Tuple[float, float, float],
                                       drone_yaw: float,
                                       ground_altitude: float = 0.0) -> Optional[Tuple[float, float, float]]:
        """
        Estimate target position assuming it's on the ground
        
        Args:
            bbox_center: (pixel_x, pixel_y) center of YOLO bounding box
            drone_position: (x, y, z) drone position in NED (z is negative altitude)
            drone_yaw: Drone heading in radians (0 = North)
            ground_altitude: Ground altitude in NED frame (usually 0.0)
        
        Returns:
            (x, y, z) estimated target position in NED, or None if cannot estimate
        """
        pixel_x, pixel_y = bbox_center
        
        # Get ray in camera frame
        ray_camera = self.pixel_to_camera_ray(pixel_x, pixel_y)
        
        # Transform to body frame
        ray_body = self.camera_to_body_frame(ray_camera)
        
        # Transform ray direction to world frame
        cos_yaw = math.cos(drone_yaw)
        sin_yaw = math.sin(drone_yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        ray_world = R @ ray_body
        
        # Ray-plane intersection (plane is ground at z = ground_altitude)
        # Ray: P = drone_pos + t * ray_world
        # Plane: z = ground_altitude
        # Solve for t:
        
        drone_z = drone_position[2]
        ray_z = ray_world[2]
        
        # Check if ray is pointing toward ground
        if ray_z <= 0:
            # Ray is horizontal or pointing up - can't hit ground
            return None
        
        # Calculate intersection parameter t
        t = (ground_altitude - drone_z) / ray_z
        
        if t < 0:
            # Intersection is behind drone
            return None
        
        # Calculate intersection point
        target_x = drone_position[0] + t * ray_world[0]
        target_y = drone_position[1] + t * ray_world[1]
        target_z = ground_altitude
        
        return (target_x, target_y, target_z)
    
    def generate_approach_waypoint(self,
                                  target_position: Tuple[float, float, float],
                                  approach_distance: float = 5.0,
                                  approach_altitude: float = -3.0) -> Tuple[float, float, float]:
        """
        Generate a waypoint to approach the target
        
        Args:
            target_position: (x, y, z) target position in NED
            approach_distance: Horizontal distance to maintain from target (meters)
            approach_altitude: Altitude to approach at (negative in NED)
        
        Returns:
            (x, y, z) waypoint position in NED
        """
        # For now, approach from directly above
        # In future, could approach from current direction
        waypoint_x = target_position[0]
        waypoint_y = target_position[1]
        waypoint_z = approach_altitude  # Fixed altitude
        
        return (waypoint_x, waypoint_y, waypoint_z)


class VisionNavigator:
    """
    High-level vision-based navigation controller
    
    Combines YOLO detections with camera geometry to generate navigation commands
    """
    
    def __init__(self):
        self.camera = CameraGeometry()
        self.last_detection: Optional[Dict] = None
        self.target_position: Optional[Tuple[float, float, float]] = None
    
    def process_detection(self,
                         bbox: Tuple[int, int, int, int],
                         class_name: str,
                         confidence: float,
                         drone_position: Tuple[float, float, float],
                         drone_yaw: float) -> Optional[Dict]:
        """
        Process a YOLO detection and generate navigation information
        
        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max) in pixels
            class_name: Detected object class
            confidence: Detection confidence (0-1)
            drone_position: Current drone position (x, y, z) in NED
            drone_yaw: Current drone heading in radians
        
        Returns:
            Dictionary with navigation info, or None if cannot process
        """
        # Calculate bounding box center
        x_min, y_min, x_max, y_max = bbox
        bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        # Estimate target position on ground
        target_pos = self.camera.estimate_target_ground_position(
            bbox_center, drone_position, drone_yaw
        )
        
        if target_pos is None:
            return None
        
        # Generate approach waypoint
        waypoint = self.camera.generate_approach_waypoint(target_pos)
        
        # Calculate distance to target
        distance_2d = math.sqrt(
            (target_pos[0] - drone_position[0])**2 +
            (target_pos[1] - drone_position[1])**2
        )
        
        self.last_detection = {
            'class': class_name,
            'confidence': confidence,
            'bbox_center': bbox_center,
            'target_position': target_pos,
            'approach_waypoint': waypoint,
            'distance': distance_2d
        }
        
        return self.last_detection


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VISION NAVIGATOR - Camera Geometry Test")
    print("=" * 70)
    
    # Initialize
    camera = CameraGeometry()
    
    # Test 1: Center of image
    print("\nTest 1: Object in center of image")
    print("-" * 70)
    
    center_x, center_y = 640, 480  # Image center
    ray = camera.pixel_to_camera_ray(center_x, center_y)
    print(f"Pixel: ({center_x}, {center_y})")
    print(f"Camera ray (normalized): {ray}")
    print(f"Ray should point forward (Z≈1): ✓" if ray[2] > 0.9 else "✗")
    
    # Test 2: Estimate ground position
    print("\nTest 2: Estimate target on ground")
    print("-" * 70)
    
    # Simulated drone state
    drone_pos = (0.0, 0.0, -5.0)  # 5 meters altitude (z is negative in NED)
    drone_yaw = 0.0  # Facing north
    
    # Object in center of image
    target = camera.estimate_target_ground_position(
        (640, 480), drone_pos, drone_yaw
    )
    
    if target:
        print(f"Drone position: {drone_pos}")
        print(f"Drone altitude: {-drone_pos[2]:.1f} m")
        print(f"Estimated target on ground: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        
        distance = math.sqrt(target[0]**2 + target[1]**2)
        print(f"Horizontal distance to target: {distance:.2f} m")
    else:
        print("Could not estimate target position")
    
    # Test 3: Complete navigation flow
    print("\nTest 3: Complete vision navigation pipeline")
    print("-" * 70)
    
    navigator = VisionNavigator()
    
    # Simulate YOLO detection
    bbox = (400, 300, 800, 700)  # Bounding box in pixels
    class_name = "person"
    confidence = 0.92
    
    result = navigator.process_detection(
        bbox, class_name, confidence, drone_pos, drone_yaw
    )
    
    if result:
        print(f"Detection: {result['class']} (confidence: {result['confidence']:.2%})")
        print(f"Target position (NED): {result['target_position']}")
        print(f"Approach waypoint (NED): {result['approach_waypoint']}")
        print(f"Distance to target: {result['distance']:.2f} m")
        print("\n✓ Vision navigation pipeline working!")
    else:
        print("✗ Could not process detection")
    
    print("\n" + "=" * 70)
    print("Next step: Integrate with YOLO detector and MAVSDK control")
    print("=" * 70)