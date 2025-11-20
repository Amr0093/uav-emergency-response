"""
Mission Controller

Main mission logic for autonomous UAV search and rescue.
Coordinates between flight control, detection, and path planning.
"""

import asyncio
import math
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from typing import Tuple, Optional
from config import DroneConfig, PathConfig
from utils.coordinate_utils import calculate_distance_3d


class MissionController:
    """
    Handles autonomous mission execution
    """
    
    def __init__(self):
        """Initialize mission controller"""
        self.drone = System()
        
        # Mission state
        self.is_running = False
        self.target_detected = False
        self.target_confirmed = False
        self.current_position = (0.0, 0.0, 0.0)
        self.current_heading = 0.0
        
        print("Mission Controller initialized")
    
    async def connect(self):
        """Connect to the drone"""
        print(f"🔌 Connecting to drone at {DroneConfig.MAVLINK_ADDRESS}...")
        await self.drone.connect(system_address=DroneConfig.MAVLINK_ADDRESS)
        
        print("⏳ Waiting for drone connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("✅ Drone connected!")
                break
    
    async def wait_for_position(self):
        """Wait for GPS fix"""
        print("📡 Waiting for GPS fix...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("✅ GPS ready!")
                break
    
    async def get_position(self) -> Tuple[float, float, float]:
        """
        Get current drone position in NED
        
        Returns:
            (north, east, down) in meters
        """
        position_ned = await self.drone.telemetry.position_velocity_ned().__anext__()
        return (
            position_ned.position.north_m,
            position_ned.position.east_m,
            position_ned.position.down_m
        )
    
    async def get_heading(self) -> float:
        """
        Get current drone heading
        
        Returns:
            Heading in radians
        """
        heading = await self.drone.telemetry.heading().__anext__()
        return math.radians(heading.heading_deg)
    
    async def update_state(self):
        """Update current position and heading"""
        self.current_position = await self.get_position()
        self.current_heading = await self.get_heading()
    
    async def arm_and_takeoff(self, altitude: float = None):
        """
        Arm and takeoff to specified altitude
        
        Args:
            altitude: Target altitude in meters (uses config default if None)
        """
        altitude = altitude or DroneConfig.TAKEOFF_ALTITUDE
        
        print(f"🚁 Arming and taking off to {altitude}m...")
        
        # Arm
        print("  Arming...")
        await self.drone.action.arm()
        
        # Set takeoff altitude and takeoff
        print(f"  Taking off to {altitude}m...")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()
        
        # Wait for altitude
        await asyncio.sleep(1)
        async for position in self.drone.telemetry.position():
            if position.relative_altitude_m >= altitude * 0.9:
                print(f"✅ Reached altitude: {position.relative_altitude_m:.1f}m")
                break
            await asyncio.sleep(0.1)
        
        # Update state
        await self.update_state()
    
    async def goto_position(self, north: float, east: float, down: float, 
                           yaw: float = 0.0, description: str = "") -> bool:
        """
        Fly to position using offboard control
        
        Args:
            north, east, down: Target position in NED (meters)
            yaw: Target heading in radians
            description: Optional description for logging
            
        Returns:
            True if reached target, False if failed/timeout
        """
        if description:
            print(f"📍 {description}")
        print(f"   Flying to: N={north:.2f}m, E={east:.2f}m, Alt={-down:.2f}m")
        
        # Set initial setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north, east, down, yaw)
        )
        
        # Start offboard mode
        try:
            await self.drone.offboard.start()
            print("   Offboard mode started")
        except OffboardError as e:
            print(f"⚠️  Offboard mode failed: {e}")
            return False
        
        # Fly to target with continuous setpoint streaming
        target_reached = False
        timeout_iterations = 300  # 30 seconds
        
        for i in range(timeout_iterations):
            # Keep sending setpoint
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(north, east, down, yaw)
            )
            
            # Check position every second
            if i % 10 == 0:
                current_pos = await self.get_position()
                
                distance = calculate_distance_3d(
                    current_pos,
                    (north, east, down)
                )
                
                print(f"   Distance to target: {distance:.2f}m")
                
                if distance < DroneConfig.POSITION_THRESHOLD:
                    target_reached = True
                    print(f"✅ Reached target (error: {distance:.2f}m)")
                    break
            
            await asyncio.sleep(0.1)
        
        # Stop offboard mode
        await self.drone.offboard.stop()
        print("   Offboard mode stopped")
        
        # Update state
        await self.update_state()
        
        return target_reached
    
    async def fly_waypoints(self, waypoints: list) -> Optional[int]:
        """
        Fly through list of waypoints
        
        Args:
            waypoints: List of (north, east, down, description) tuples
            
        Returns:
            Index of waypoint where target was detected, or None
        """
        print(f"\n📹 Flying {len(waypoints)} waypoints...")
        
        for i, waypoint in enumerate(waypoints, 1):
            north, east, down, description = waypoint
            
            print(f"\n   Waypoint {i}/{len(waypoints)}: {description}")
            
            success = await self.goto_position(
                north, east, down, 
                self.current_heading,
                f"Waypoint {i}"
            )
            
            if success:
                print(f"   ✅ Reached waypoint {i}")
                
                # Pause for detection check (if integrated with camera)
                await asyncio.sleep(1)
                
                # Check if target detected (placeholder for now)
                # In full system, this would check detection results
                if i == 2:  # Simulate detection at waypoint 2
                    print(f"\n   🎯 TARGET DETECTED at waypoint {i}!")
                    self.target_detected = True
                    return i
            else:
                print(f"   ⚠️  Failed to reach waypoint {i}")
        
        return None
    
    async def return_and_land(self):
        """Return to launch and land"""
        print("\n🏠 Returning to home...")
        await self.drone.action.return_to_launch()
        
        print("⏬ Landing...")
        await asyncio.sleep(10)  # Wait for RTL to complete
    
    async def run_search_mission(self):
        """
        Execute complete search mission
        """
        self.is_running = True
        
        print("\n" + "="*70)
        print("🚁 AUTONOMOUS SEARCH & RESCUE MISSION")
        print("="*70 + "\n")
        
        try:
            # 1. Connect and prepare
            await self.connect()
            await self.wait_for_position()
            
            # 2. Takeoff
            await self.arm_and_takeoff()
            await asyncio.sleep(2)
            
            # 3. Fly search pattern
            waypoints = [
                (wp[0], wp[1], wp[2], wp[3]) 
                for wp in PathConfig.SEARCH_WAYPOINTS
            ]
            
            detection_waypoint = await self.fly_waypoints(waypoints)
            
            if detection_waypoint is not None:
                print(f"\n✅ Target confirmed at waypoint {detection_waypoint}!")
                self.target_confirmed = True
            else:
                print("\n⚠️  Search complete - no target found")
            
            # 4. Return and land
            await self.return_and_land()
            
            print("\n✅ Mission complete!")
            
        except Exception as e:
            print(f"\n❌ Mission failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_running = False


# Main entry point
async def main():
    """Run mission"""
    controller = MissionController()
    await controller.run_search_mission()


if __name__ == "__main__":
    print("Starting mission controller...")
    print("Make sure PX4 SITL + Gazebo is running!\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Mission cancelled by user")
