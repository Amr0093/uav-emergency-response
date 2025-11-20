"""
Coordinate and geometry utility functions

Handles conversions between different coordinate systems and
geometric calculations for UAV navigation.
"""

import math
from typing import Tuple


def ned_to_gps(north: float, east: float, down: float, 
               home_lat: float, home_lon: float, home_alt: float) -> Tuple[float, float, float]:
    """
    Convert NED (North-East-Down) coordinates to GPS coordinates
    
    Args:
        north: North offset in meters
        east: East offset in meters  
        down: Down offset in meters (negative = up)
        home_lat: Home latitude in degrees
        home_lon: Home longitude in degrees
        home_alt: Home altitude in meters
        
    Returns:
        Tuple of (latitude, longitude, altitude)
    """
    # Earth radius in meters
    EARTH_RADIUS = 6371000.0
    
    # Convert offsets to degrees
    d_lat = north / EARTH_RADIUS * (180.0 / math.pi)
    d_lon = east / (EARTH_RADIUS * math.cos(math.pi * home_lat / 180.0)) * (180.0 / math.pi)
    
    # Calculate new GPS position
    new_lat = home_lat + d_lat
    new_lon = home_lon + d_lon
    new_alt = home_alt - down  # NED down is negative of altitude
    
    return (new_lat, new_lon, new_alt)


def calculate_distance_3d(pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
    """
    Calculate 3D Euclidean distance between two points
    
    Args:
        pos1: First position (x, y, z)
        pos2: Second position (x, y, z)
        
    Returns:
        Distance in meters
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    
    return math.sqrt(dx**2 + dy**2 + dz**2)


def calculate_bearing(pos1: Tuple[float, float], 
                      pos2: Tuple[float, float]) -> float:
    """
    Calculate bearing (heading) from pos1 to pos2 in radians
    
    Args:
        pos1: Start position (x, y) in NED
        pos2: End position (x, y) in NED
        
    Returns:
        Bearing in radians (0 = North, π/2 = East)
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    bearing = math.atan2(dy, dx)
    return bearing


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in radians
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    
    return angle


def pixel_to_world_ray(pixel_x: int, pixel_y: int, 
                       camera_matrix: dict, 
                       drone_pose: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    Convert pixel coordinates to a ray in world coordinates
    
    This creates a ray from camera through the pixel point.
    Useful for estimating target position from camera image.
    
    Args:
        pixel_x: Pixel x coordinate
        pixel_y: Pixel y coordinate
        camera_matrix: Dict with 'fx', 'fy', 'cx', 'cy'
        drone_pose: (x, y, z, yaw) of drone in NED
        
    Returns:
        Ray direction as (dx, dy, dz) unit vector
    """
    # Normalize pixel coordinates (from camera center)
    x_norm = (pixel_x - camera_matrix['cx']) / camera_matrix['fx']
    y_norm = (pixel_y - camera_matrix['cy']) / camera_matrix['fy']
    
    # Ray in camera frame (Z forward, X right, Y down)
    ray_camera = (x_norm, y_norm, 1.0)
    
    # Rotate to world frame using drone yaw
    yaw = drone_pose[3]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    
    # Simple 2D rotation for now (assuming camera points down)
    ray_world_x = ray_camera[0] * cos_yaw - ray_camera[1] * sin_yaw
    ray_world_y = ray_camera[0] * sin_yaw + ray_camera[1] * cos_yaw
    ray_world_z = ray_camera[2]
    
    # Normalize to unit vector
    magnitude = math.sqrt(ray_world_x**2 + ray_world_y**2 + ray_world_z**2)
    
    return (ray_world_x / magnitude, 
            ray_world_y / magnitude, 
            ray_world_z / magnitude)
