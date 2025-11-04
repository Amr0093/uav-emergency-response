"""
A* Pathfinding Algorithm for 2D Grid
For UAV Emergency Response System

Author: Amr Hassan
Date: November 2024
"""

import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set


class Node:
    """Represents a cell in the grid for A* algorithm"""
    
    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position  # (row, col)
        self.parent = parent      # Parent node in the path
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current to goal
        self.f = 0  # Total cost (g + h)
    
    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        """For heap comparison - compare by f score"""
        return self.f < other.f
    
    def __hash__(self):
        return hash(self.position)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two positions.
    Perfect for grid-based movement (no diagonal).
    
    Args:
        pos1: First position (row, col)
        pos2: Second position (row, col)
    
    Returns:
        Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(position: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get valid neighboring positions (up, down, left, right).
    
    Args:
        position: Current position (row, col)
        grid: 2D numpy array where 0=free, 1=obstacle
    
    Returns:
        List of valid neighbor positions
    """
    row, col = position
    rows, cols = grid.shape
    neighbors = []
    
    # Define 4-directional movement (no diagonal)
    directions = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1)    # Right
    ]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        
        # Check if neighbor is within grid bounds
        if 0 <= new_row < rows and 0 <= new_col < cols:
            # Check if neighbor is not an obstacle
            if grid[new_row, new_col] == 0:
                neighbors.append((new_row, new_col))
    
    return neighbors


def reconstruct_path(node: Node) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from start to goal by following parent pointers.
    
    Args:
        node: The goal node
    
    Returns:
        List of positions from start to goal
    """
    path = []
    current = node
    
    while current is not None:
        path.append(current.position)
        current = current.parent
    
    path.reverse()  # Reverse to get start -> goal
    return path


def a_star(grid: np.ndarray, 
           start: Tuple[int, int], 
           goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm on a 2D grid.
    
    Args:
        grid: 2D numpy array where 0=free, 1=obstacle
        start: Starting position (row, col)
        goal: Goal position (row, col)
    
    Returns:
        List of positions from start to goal, or None if no path exists
    """
    
    # Validate start and goal
    if grid[start[0], start[1]] == 1:
        print("Error: Start position is an obstacle!")
        return None
    if grid[goal[0], goal[1]] == 1:
        print("Error: Goal position is an obstacle!")
        return None
    
    # Initialize start node
    start_node = Node(start)
    start_node.g = 0
    start_node.h = manhattan_distance(start, goal)
    start_node.f = start_node.h
    
    # Initialize goal node
    goal_node = Node(goal)
    
    # Open set (nodes to be evaluated) - using heap for efficiency
    open_set = []
    heapq.heappush(open_set, start_node)
    
    # Closed set (already evaluated nodes)
    closed_set: Set[Tuple[int, int]] = set()
    
    # Keep track of best g_score for each position
    g_scores = {start: 0}
    
    while open_set:
        # Get node with lowest f score
        current_node = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        if current_node.position == goal:
            return reconstruct_path(current_node)
        
        # Add to closed set
        closed_set.add(current_node.position)
        
        # Explore neighbors
        for neighbor_pos in get_neighbors(current_node.position, grid):
            
            # Skip if already evaluated
            if neighbor_pos in closed_set:
                continue
            
            # Create neighbor node
            neighbor_node = Node(neighbor_pos, current_node)
            
            # Calculate g score (cost from start to neighbor)
            # Moving one cell = cost of 1
            tentative_g = current_node.g + 1
            
            # Check if this path to neighbor is better than any previous one
            if tentative_g < g_scores.get(neighbor_pos, float('inf')):
                # This path is the best so far - record it
                neighbor_node.parent = current_node
                neighbor_node.g = tentative_g
                neighbor_node.h = manhattan_distance(neighbor_pos, goal)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                g_scores[neighbor_pos] = tentative_g
                
                # Add to open set
                heapq.heappush(open_set, neighbor_node)
    
    # Open set is empty but goal was never reached
    print("No path found!")
    return None


def create_grid(rows: int, cols: int, obstacle_probability: float = 0.3) -> np.ndarray:
    """
    Create a grid with random obstacles.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        obstacle_probability: Probability of each cell being an obstacle (0.0 to 1.0)
    
    Returns:
        2D numpy array where 0=free, 1=obstacle
    """
    grid = np.zeros((rows, cols), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_probability:
                grid[i, j] = 1
    
    return grid


def visualize_path(grid: np.ndarray, 
                   path: Optional[List[Tuple[int, int]]], 
                   start: Tuple[int, int], 
                   goal: Tuple[int, int],
                   save_path: str = None):
    """
    Visualize the grid, obstacles, and path using matplotlib.
    
    Args:
        grid: 2D numpy array
        path: List of positions in the path (or None if no path)
        start: Start position
        goal: Goal position
        save_path: Optional file path to save the figure
    """
    plt.figure(figsize=(10, 10))
    
    # Create a colored version of the grid for better visualization
    # 0 = white (free), 1 = black (obstacle)
    colored_grid = np.copy(grid).astype(float)
    
    # Display the grid
    plt.imshow(colored_grid, cmap='binary', interpolation='nearest')
    
    # Draw path if it exists
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        plt.plot(path_cols, path_rows, 'b-', linewidth=3, label=f'Path (length: {len(path)})')
        plt.plot(path_cols, path_rows, 'b.', markersize=8)
    
    # Mark start and goal
    plt.plot(start[1], start[0], 'go', markersize=20, label='Start', markeredgecolor='black', markeredgewidth=2)
    plt.plot(goal[1], goal[0], 'ro', markersize=20, label='Goal', markeredgecolor='black', markeredgewidth=2)
    
    # Add grid lines
    plt.grid(True, alpha=0.3)
    
    # Labels and title
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    plt.title('A* Pathfinding on 2D Grid', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    
    # Add color bar to show what colors mean
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='binary'), ax=plt.gca())
    cbar.set_label('0=Free, 1=Obstacle', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("A* PATHFINDING ALGORITHM - 2D GRID")
    print("UAV Emergency Response System")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # ========== CONFIGURATION ==========
    GRID_ROWS = 20
    GRID_COLS = 20
    OBSTACLE_PROB = 0.25  # 25% of cells will be obstacles
    
    START = (1, 1)        # Top-left area
    GOAL = (18, 18)       # Bottom-right area
    
    # ========== CREATE GRID ==========
    print(f"\nCreating {GRID_ROWS}x{GRID_COLS} grid...")
    grid = create_grid(GRID_ROWS, GRID_COLS, OBSTACLE_PROB)
    
    # Ensure start and goal are not obstacles
    grid[START[0], START[1]] = 0
    grid[GOAL[0], GOAL[1]] = 0
    
    obstacle_count = np.sum(grid)
    print(f"Grid created with {obstacle_count} obstacles ({obstacle_count/(GRID_ROWS*GRID_COLS)*100:.1f}%)")
    
    # ========== RUN A* ==========
    print(f"\nSearching for path from {START} to {GOAL}...")
    print("Running A* algorithm...")
    
    path = a_star(grid, START, GOAL)
    
    # ========== RESULTS ==========
    if path:
        print(f"\n✓ Path found!")
        print(f"  Path length: {len(path)} steps")
        print(f"  Path cost: {len(path) - 1} (each move costs 1)")
        print(f"\n  First 5 steps: {path[:5]}")
        if len(path) > 5:
            print(f"  Last 5 steps: {path[-5:]}")
    else:
        print("\n✗ No path found!")
        print("  Try reducing obstacle probability or changing start/goal positions")
    
    # ========== VISUALIZE ==========
    print("\nGenerating visualization...")
    visualize_path(grid, path, START, GOAL)
    
    print("\n" + "=" * 60)
    print("A* Algorithm Explanation:")
    print("=" * 60)
    print("f(n) = g(n) + h(n)")
    print("  g(n) = actual cost from start to current node")
    print("  h(n) = estimated cost from current to goal (Manhattan distance)")
    print("  f(n) = total estimated cost of path through node n")
    print("\nA* always expands the node with lowest f(n) first.")
    print("This guarantees finding the shortest path!")
    print("=" * 60)