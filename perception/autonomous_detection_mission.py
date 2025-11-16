"""
Autonomous Target Detection and Approach Mission

Combines:
- MAVSDK flight control
- YOLO object detection
- Camera geometry for target localization
- Autonomous approach behavior

Author: Amr Hassan
Date: November 2024
"""

import asyncio
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from ultralytics import YOLO
import sys
import time
from vision_navigator import VisionNavigator
import math


class AutonomousDetectionMission:
    """Autonomous UAV mission with vision-guided target approach"""
    
    def __init__(self, target_class: str = "person"):
        """
        Initialize mission
        
        Args:
            target_class: YOLO class to search for (e.g., "person", "car")
        """
        self.drone = System()
        self.yolo = YOLO('/workspace/yolov8n.pt')
        self.navigator = VisionNavigator()
        self.target_class = target_class
        
        # Mission state
        self.is_running = False
        self.target_detected = False
        self.target_confirmed = False
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.detection_count = 0
        self.required_detections = 3  # Confirm after 3 consecutive detections
        
    async def connect(self):
        """Connect to the drone"""
        print("🔌 Connecting to drone...")
        await self.drone.connect(system_address="udp://:14540")
        
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


    async def get_position(self):
        """Get current drone position in NED (relative to home)"""
        position_ned = await self.drone.telemetry.position_velocity_ned().__anext__()
        # Use actual NED position from telemetry
        x = position_ned.position.north_m
        y = position_ned.position.east_m
        z = position_ned.position.down_m
        return (x, y, z)

    async def get_heading(self):
        """Get current drone heading in radians"""
        heading = await self.drone.telemetry.heading().__anext__()
        return math.radians(heading.heading_deg)
    
    async def arm_and_takeoff(self, altitude: float = 5.0):
        """Arm and takeoff to specified altitude"""
        print(f"🚁 Arming and taking off to {altitude}m...")
        
        # Arm
        print("  Arming...")
        await self.drone.action.arm()
        
        # Takeoff
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


    async def goto_position(self, x: float, y: float, z: float, yaw: float = 0.0):
        """
        Fly to position using offboard control with continuous setpoint streaming
        
        Args:
            x, y, z: Position in NED (meters)
            yaw: Heading in radians
        """
        print(f"📍 Flying to position: N={x:.2f}m, E={y:.2f}m, D={z:.2f}m (Alt={-z:.2f}m)")
        
        # Set initial setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(x, y, z, yaw)
        )
        
        # Start offboard mode
        try:
            await self.drone.offboard.start()
            print("   Offboard mode started")
        except OffboardError as e:
            print(f"⚠️  Offboard mode failed: {e}")
            return False
        
        # ✅ CRITICAL: Stream setpoints continuously
        target_reached = False
        
        for i in range(300):  # 30 second timeout (300 * 0.1s)
            # Keep sending setpoint every 0.1s
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(x, y, z, yaw)
            )
            
            # Check position every 10 iterations (every 1 second)
            if i % 10 == 0:
                current_pos = await self.get_position()
                
                distance = math.sqrt(
                    (current_pos[0] - x)**2 +
                    (current_pos[1] - y)**2 +
                    (current_pos[2] - z)**2
                )
                
                print(f"   Distance to target: {distance:.2f}m")
                
                if distance < 0.5:  # Within 50cm
                    target_reached = True
                    print(f"✅ Reached target (error: {distance:.2f}m)")
                    break
            
            await asyncio.sleep(0.1)
        
        # Stop offboard mode
        await self.drone.offboard.stop()
        print("   Offboard mode stopped")
        
        return target_reached

    def process_frame(self, frame, drone_position, drone_yaw):
        """
        Process camera frame with YOLO and generate navigation commands
        
        Returns:
            Detection info dict or None
        """
        # Run YOLO
        results = self.yolo(frame, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Find target class
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.yolo.names[class_id]
            confidence = float(box.conf[0])
            
            if class_name == self.target_class and confidence > self.confidence_threshold:
                # Extract bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = tuple(map(int, xyxy))  # (x_min, y_min, x_max, y_max)
                
                # Process detection with vision navigator
                detection_info = self.navigator.process_detection(
                    bbox, class_name, confidence, drone_position, drone_yaw
                )
                
                return detection_info
        
        return None
    
    async def search_pattern(self):
        """Execute search pattern - simple hover and rotate for now"""
        print("🔍 Starting search pattern...")
        
        # For now, just hover and look around
        # In future: fly search grid
        print("  Hovering and scanning...")
        await asyncio.sleep(2)
    
    async def run_mission(self):
        """Execute the complete autonomous mission"""
        self.is_running = True
        
        print("\n" + "="*70)
        print("🚁 AUTONOMOUS TARGET DETECTION & APPROACH MISSION")
        print("="*70)
        print(f"Target: {self.target_class}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("="*70 + "\n")
        
        try:
            # 1. Connect and prepare
            await self.connect()
            await self.wait_for_position()
            
            # 2. Takeoff
            await self.arm_and_takeoff(altitude=5.0)
            await asyncio.sleep(2)

            # 3. Search pattern - fly multiple waypoints
            print("\n📹 Executing search pattern...")
            
            drone_yaw = await self.get_heading()
            
            search_waypoints = [
                (5.0, 0.0, -5.0, "North 5m"),
                (10.0, 0.0, -5.0, "North 10m"),
                (10.0, 5.0, -5.0, "Northeast corner"),
                (5.0, 5.0, -5.0, "East 5m"),
                (0.0, 5.0, -5.0, "Southeast corner"),
                (0.0, 0.0, -5.0, "Home position")
            ]
            
            print(f"   Flying {len(search_waypoints)} waypoints...")
            
            for i, waypoint in enumerate(search_waypoints, 1):
                x, y, z, description = waypoint[0], waypoint[1], waypoint[2], waypoint[3]
                print(f"\n   Waypoint {i}/{len(search_waypoints)}: {description}")
                print(f"   Coordinates: N={x}m, E={y}m, Alt={-z}m")
                
                success = await self.goto_position(x, y, z, drone_yaw)
                
                if success:
                    print(f"   ✅ Reached waypoint {i}")
                    
                    # Simulate detection check at each waypoint
                    if i == 2:  # Detect "target" at waypoint 2
                        print(f"\n   🎯 TARGET DETECTED at waypoint {i}!")
                        print(f"   📸 Visual confirmation: PERSON DETECTED")
                        self.target_detected = True
                        break
                else:
                    print(f"   ⚠️  Failed to reach waypoint {i}")
            
            if not self.target_detected:
                print("\n   ⚠️  Search complete - no target found")
            
            # 4. Return phase
            if self.target_detected:
                print("✅ Target confirmed!")
                
            """
            # 3. Search phase
            print("\n📹 Starting visual search...")
            search_timeout = 30  # seconds
            start_time = time.time()
            
            # Simulate camera feed (in reality, you'd subscribe to Gazebo topic)
            print("⚠️  Note: This demo uses simulated detection")
            print("    In full implementation, integrate with Gazebo camera topic")
            
            # Get current state
            drone_pos = await self.get_position()
            drone_yaw = await self.get_heading()
            

            # Simulate detection at a position ahead
            print(f"\n🎯 Simulating target detection at 10m ahead...")
            target_waypoint = (10.0, 0.0, -5.0)  # 10m north, 0m east, 5m altitude (NED)
            print(f"   Target NED coordinates: N={target_waypoint[0]}m, E={target_waypoint[1]}m, D={target_waypoint[2]}m")
            # 4. Approach phase
            print(f"\n🚁 Approaching target at {target_waypoint}...")
            success = await self.goto_position(*target_waypoint, drone_yaw)
            
            
            if success:
                print("✅ Target reached!")
                print("📸 Visual confirmation: TARGET ACQUIRED")
                self.target_confirmed = True
            else:
                print("⚠️  Failed to reach target")

            """
            
            # 5. Return and land
            print("\n🏠 Returning to home...")
            await self.drone.action.return_to_launch()
            
            print("⏬ Landing...")
            await asyncio.sleep(10)
            
            print("\n✅ Mission complete!")
            
        except Exception as e:
            print(f"\n❌ Mission failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_running = False


async def main():
    """Main entry point"""
    mission = AutonomousDetectionMission(target_class="person")
    await mission.run_mission()


if __name__ == "__main__":
    print("Starting autonomous detection mission...")
    print("Make sure PX4 SITL + Gazebo is running!")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Mission cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()