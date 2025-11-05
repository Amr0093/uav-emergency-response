"""
RRT* (Rapidly-exploring Random Tree Star) Algorithm for 2D Space
For UAV Emergency Response System

RRT* is a sampling-based path planning algorithm that:
1. Randomly samples the configuration space
2. Grows a tree from start toward goal
3. Rewires the tree to optimize path cost (the * improvement)
4. Works well in continuous, high-dimensional spaces

Author: Amr Hassan
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import random
import math


class Node:
    """Represents a node in the RRT* tree"""
    
    def __init__(self, position: Tuple[float, float]):
        self.position = position  # (x, y) in continuous space
        self.parent: Optional[Node] = None
        self.cost = 0.0  # Cost from start to this node
        self.children: List[Node] = []
    
    def __repr__(self):
        return f"Node({self.position}, cost={self.cost:.2f})"


class RRTStar:
    """
    RRT* path planning algorithm
    
    Key differences from basic RRT:
    1. Choose parent: Select best parent within radius (not just nearest)
    2. Rewire: After adding node, check if it provides better path to nearby nodes
    """
    
    def __init__(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                 obstacles: List[Tuple[float, float, float, float]],
                 max_iterations: int = 3000,
                 step_size: float = 1.0,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 3.0):
        """
        Initialize RRT* planner
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            bounds: ((x_min, x_max), (y_min, y_max))
            obstacles: List of rectangles (x, y, width, height)
            max_iterations: Maximum number of iterations
            step_size: Maximum distance to extend tree in one step
            goal_sample_rate: Probability of sampling goal (vs random point)
            search_radius: Radius for choosing parent and rewiring
        """
        self.start = Node(start)
        self.goal = Node(goal)
        self.bounds = bounds
        self.obstacles = obstacles
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        
        # Tree starts with just the start node
        self.nodes = [self.start]
        self.goal_node: Optional[Node] = None
        
        # For visualization
        self.edges = []  # Store all tree edges for drawing
    
    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """
        Execute RRT* planning algorithm
        
        Returns:
            List of positions from start to goal, or None if no path found
        """
        print(f"Starting RRT* planning...")
        print(f"  Start: {self.start.position}")
        print(f"  Goal: {self.goal.position}")
        print(f"  Max iterations: {self.max_iterations}")
        
        for i in range(self.max_iterations):
            # Sample random point (or goal with some probability)
            random_point = self._sample()
            
            # Find nearest node in tree
            nearest_node = self._get_nearest_node(random_point)
            
            # Steer toward random point (limit by step_size)
            new_position = self._steer(nearest_node.position, random_point)
            
            # Check if path from nearest to new is collision-free
            if not self._is_collision_free(nearest_node.position, new_position):
                continue
            
            # Create new node
            new_node = Node(new_position)
            
            # Find nearby nodes within search radius
            nearby_nodes = self._get_nearby_nodes(new_node, self.search_radius)
            
            # Choose best parent (RRT* improvement #1)
            best_parent = self._choose_parent(new_node, nearest_node, nearby_nodes)
            
            if best_parent is None:
                continue  # No valid parent found
            
            # Add new node to tree
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + self._distance(best_parent.position, new_node.position)
            best_parent.children.append(new_node)
            self.nodes.append(new_node)
            self.edges.append((best_parent.position, new_node.position))
            
            # Rewire tree (RRT* improvement #2)
            self._rewire(new_node, nearby_nodes)
            
            # Check if we can connect to goal
            if self._distance(new_node.position, self.goal.position) <= self.step_size:
                if self._is_collision_free(new_node.position, self.goal.position):
                    # Found path to goal!
                    self.goal_node = Node(self.goal.position)
                    self.goal_node.parent = new_node
                    self.goal_node.cost = new_node.cost + self._distance(new_node.position, self.goal.position)
                    new_node.children.append(self.goal_node)
                    self.nodes.append(self.goal_node)
                    self.edges.append((new_node.position, self.goal_node.position))
                    print(f"\n✓ Path found at iteration {i+1}!")
                    # Continue for a bit to optimize
                    if i > 500:  # Found early, keep optimizing
                        continue
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Iteration {i+1}/{self.max_iterations} - Tree size: {len(self.nodes)}")
        
        if self.goal_node is None:
            print("\n✗ No path found after maximum iterations")
            return None
        
        # Extract path
        path = self._extract_path()
        print(f"  Final path cost: {self.goal_node.cost:.2f}")
        print(f"  Final tree size: {len(self.nodes)} nodes")
        return path
    
    def _sample(self) -> Tuple[float, float]:
        """Sample a random point in the space (or goal with some probability)"""
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        
        x = random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = random.uniform(self.bounds[1][0], self.bounds[1][1])
        return (x, y)
    
    def _get_nearest_node(self, position: Tuple[float, float]) -> Node:
        """Find the nearest node in the tree to the given position"""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            dist = self._distance(node.position, position)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Steer from from_pos toward to_pos, limited by step_size
        
        Returns new position that is at most step_size away from from_pos
        """
        dist = self._distance(from_pos, to_pos)
        
        if dist <= self.step_size:
            return to_pos
        
        # Move step_size in direction of to_pos
        theta = math.atan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])
        new_x = from_pos[0] + self.step_size * math.cos(theta)
        new_y = from_pos[1] + self.step_size * math.sin(theta)
        
        return (new_x, new_y)
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _is_collision_free(self, from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> bool:
        """
        Check if line segment from from_pos to to_pos is collision-free
        
        Uses simple line-rectangle intersection check
        """
        # Check multiple points along the line
        steps = int(self._distance(from_pos, to_pos) / 0.1) + 1
        
        for i in range(steps + 1):
            t = i / steps
            x = from_pos[0] + t * (to_pos[0] - from_pos[0])
            y = from_pos[1] + t * (to_pos[1] - from_pos[1])
            
            # Check if point is inside any obstacle
            for obs in self.obstacles:
                obs_x, obs_y, obs_w, obs_h = obs
                if (obs_x <= x <= obs_x + obs_w and
                    obs_y <= y <= obs_y + obs_h):
                    return False
        
        return True
    
    def _get_nearby_nodes(self, node: Node, radius: float) -> List[Node]:
        """Get all nodes within radius of the given node"""
        nearby = []
        for n in self.nodes:
            if self._distance(n.position, node.position) <= radius:
                nearby.append(n)
        return nearby
    
    def _choose_parent(self, 
                       new_node: Node, 
                       nearest_node: Node, 
                       nearby_nodes: List[Node]) -> Optional[Node]:
        """
        Choose the best parent for new_node from nearby nodes
        
        This is a key RRT* improvement: instead of just connecting to nearest,
        we check all nearby nodes and choose the one that gives lowest cost
        """
        best_parent = nearest_node
        min_cost = nearest_node.cost + self._distance(nearest_node.position, new_node.position)
        
        for node in nearby_nodes:
            # Calculate cost if we use this node as parent
            potential_cost = node.cost + self._distance(node.position, new_node.position)
            
            # Check if this is better AND collision-free
            if potential_cost < min_cost:
                if self._is_collision_free(node.position, new_node.position):
                    best_parent = node
                    min_cost = potential_cost
        
        return best_parent
    
    def _rewire(self, new_node: Node, nearby_nodes: List[Node]):
        """
        Rewire the tree: check if going through new_node gives better path to nearby nodes
        
        This is the second key RRT* improvement: after adding a node, we check
        if it provides a better path to any of its neighbors
        """
        for node in nearby_nodes:
            if node == new_node or node == self.start:
                continue
            
            # Calculate cost if we rewire through new_node
            new_cost = new_node.cost + self._distance(new_node.position, node.position)
            
            # If this is better than current cost, rewire
            if new_cost < node.cost:
                if self._is_collision_free(new_node.position, node.position):
                    # Remove old parent connection
                    if node.parent:
                        node.parent.children.remove(node)
                    
                    # Set new parent
                    node.parent = new_node
                    node.cost = new_cost
                    new_node.children.append(node)
                    
                    # Update costs of all descendants
                    self._update_descendants_cost(node)
    
    def _update_descendants_cost(self, node: Node):
        """Recursively update cost of all descendants after rewiring"""
        for child in node.children:
            child.cost = node.cost + self._distance(node.position, child.position)
            self._update_descendants_cost(child)
    
    def _extract_path(self) -> List[Tuple[float, float]]:
        """Extract path from start to goal by following parent pointers"""
        if self.goal_node is None:
            return []
        
        path = []
        current = self.goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path


def visualize_rrt_star(rrt: RRTStar, 
                       path: Optional[List[Tuple[float, float]]] = None,
                       show_tree: bool = True,
                       save_path: str = None):
    """
    Visualize RRT* planning result
    
    Args:
        rrt: RRTStar object after planning
        path: Final path (if found)
        show_tree: Whether to show the full search tree
        save_path: Optional file path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw obstacles
    for obs in rrt.obstacles:
        x, y, w, h = obs
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=1, 
                                 edgecolor='black', 
                                 facecolor='gray',
                                 alpha=0.7,
                                 label='Obstacle' if obs == rrt.obstacles[0] else '')
        ax.add_patch(rect)
    
    # Draw tree (all edges explored)
    if show_tree and rrt.edges:
        for edge in rrt.edges:
            from_pos, to_pos = edge
            ax.plot([from_pos[0], to_pos[0]], 
                   [from_pos[1], to_pos[1]], 
                   'c-', linewidth=0.3, alpha=0.4)
        
        # Add tree label
        ax.plot([], [], 'c-', linewidth=2, alpha=0.4, label=f'RRT* Tree ({len(rrt.nodes)} nodes)')
    
    # Draw final path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=3, label=f'Final Path (cost: {rrt.goal_node.cost:.1f})')
        ax.plot(path_x, path_y, 'b.', markersize=6)
    
    # Mark start and goal
    ax.plot(rrt.start.position[0], rrt.start.position[1], 
           'go', markersize=15, label='Start', 
           markeredgecolor='black', markeredgewidth=2)
    ax.plot(rrt.goal.position[0], rrt.goal.position[1], 
           'ro', markersize=15, label='Goal',
           markeredgecolor='black', markeredgewidth=2)
    
    # Set bounds and labels
    ax.set_xlim(rrt.bounds[0][0] - 1, rrt.bounds[0][1] + 1)
    ax.set_ylim(rrt.bounds[1][0] - 1, rrt.bounds[1][1] + 1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('RRT* (Rapidly-exploring Random Tree Star) Path Planning', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def create_random_obstacles(bounds: Tuple[Tuple[float, float], Tuple[float, float]], 
                           num_obstacles: int = 8,
                           min_size: float = 1.0,
                           max_size: float = 4.0) -> List[Tuple[float, float, float, float]]:
    """
    Create random rectangular obstacles
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max))
        num_obstacles: Number of obstacles to create
        min_size: Minimum obstacle size
        max_size: Maximum obstacle size
    
    Returns:
        List of obstacles (x, y, width, height)
    """
    obstacles = []
    x_range = bounds[0][1] - bounds[0][0]
    y_range = bounds[1][1] - bounds[1][0]
    
    for _ in range(num_obstacles):
        width = random.uniform(min_size, max_size)
        height = random.uniform(min_size, max_size)
        x = random.uniform(bounds[0][0], bounds[0][1] - width)
        y = random.uniform(bounds[1][0], bounds[1][1] - height)
        obstacles.append((x, y, width, height))
    
    return obstacles


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RRT* (Rapidly-exploring Random Tree Star) PATHFINDING ALGORITHM")
    print("UAV Emergency Response System")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # ========== CONFIGURATION ==========
    START = (2.0, 2.0)
    GOAL = (18.0, 18.0)
    BOUNDS = ((0.0, 20.0), (0.0, 20.0))  # ((x_min, x_max), (y_min, y_max))
    
    # Create obstacles (keep start and goal clear)
    print("\nGenerating obstacles...")
    obstacles = create_random_obstacles(BOUNDS, num_obstacles=10)
    
    # Ensure start and goal are not in obstacles
    obstacles = [obs for obs in obstacles 
                 if not (obs[0] <= START[0] <= obs[0] + obs[2] and 
                        obs[1] <= START[1] <= obs[1] + obs[3])]
    obstacles = [obs for obs in obstacles 
                 if not (obs[0] <= GOAL[0] <= obs[0] + obs[2] and 
                        obs[1] <= GOAL[1] <= obs[1] + obs[3])]
    
    print(f"Created {len(obstacles)} obstacles")
    
    # ========== CREATE RRT* PLANNER ==========
    rrt_star = RRTStar(
        start=START,
        goal=GOAL,
        bounds=BOUNDS,
        obstacles=obstacles,
        max_iterations=3000,
        step_size=1.0,
        goal_sample_rate=0.1,  # 10% chance to sample goal
        search_radius=3.0  # Search radius for parent selection and rewiring
    )
    
    # ========== RUN PLANNING ==========
    print("\n" + "=" * 70)
    path = rrt_star.plan()
    print("=" * 70)
    
    # ========== RESULTS ==========
    if path:
        print(f"\n✓ Path found!")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Path cost: {rrt_star.goal_node.cost:.2f}")
        print(f"  Tree size: {len(rrt_star.nodes)} nodes explored")
        print(f"\n  First 3 waypoints: {path[:3]}")
        if len(path) > 3:
            print(f"  Last 3 waypoints: {path[-3:]}")
    else:
        print("\n✗ No path found!")
        print("  Try increasing max_iterations or adjusting parameters")
    
    # ========== VISUALIZE ==========
    print("\nGenerating visualization...")
    visualize_rrt_star(rrt_star, path, show_tree=True)
    
    # ========== ALGORITHM EXPLANATION ==========
    print("\n" + "=" * 70)
    print("RRT* Algorithm Explanation:")
    print("=" * 70)
    print("RRT* builds a tree by randomly sampling the space and extending")
    print("toward samples. Key improvements over basic RRT:")
    print()
    print("1. CHOOSE PARENT: When adding a new node, check all nearby nodes")
    print("   and choose the one that gives the lowest cost path.")
    print()
    print("2. REWIRE: After adding a node, check if it provides a better")
    print("   path to any of its neighbors. If yes, rewire the tree.")
    print()
    print("These improvements make RRT* asymptotically optimal!")
    print("(As iterations → ∞, path → optimal path)")
    print("=" * 70)
    
    # ========== COMPARISON WITH A* ==========
    print("\n" + "=" * 70)
    print("RRT* vs A* Comparison:")
    print("=" * 70)
    print("RRT* ADVANTAGES:")
    print("  ✓ Works in continuous, high-dimensional spaces")
    print("  ✓ No need for discretization")
    print("  ✓ Handles complex obstacles naturally")
    print("  ✓ Probabilistically complete and asymptotically optimal")
    print()
    print("RRT* DISADVANTAGES:")
    print("  ✗ Path quality depends on number of iterations")
    print("  ✗ Not deterministic (different run = different path)")
    print("  ✗ Slower than A* on small grids")
    print("  ✗ May need post-processing (path smoothing)")
    print()
    print("A* ADVANTAGES:")
    print("  ✓ Guarantees shortest path (if found)")
    print("  ✓ Deterministic")
    print("  ✓ Fast on small/medium grids")
    print()
    print("A* DISADVANTAGES:")
    print("  ✗ Requires grid discretization")
    print("  ✗ Memory intensive in high dimensions")
    print("  ✗ Struggles in continuous spaces")
    print("=" * 70)
    
    print("\nUSE RRT* WHEN:")
    print("  • Working in continuous space (like real UAV flight)")
    print("  • High-dimensional problems (3D + orientation)")
    print("  • Complex obstacle geometries")
    print("  • Don't need exact optimal path")
    print()
    print("USE A* WHEN:")
    print("  • Grid-based environment")
    print("  • Need guaranteed shortest path")
    print("  • 2D or simple 3D space")
    print("  • Real-time performance critical")
    print("=" * 70)