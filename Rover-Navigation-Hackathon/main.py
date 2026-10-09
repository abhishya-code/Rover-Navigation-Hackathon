"""
LunarRover AI: Autonomous Navigation and Obstacle Avoidance System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import ensure_directories
from terrain_processing import TerrainProcessor
from obstacle_detection import ObstacleDetector
from occupancy_grid import OccupancyGrid
from rover_simulation import RoverSimulation

def find_nearest_free_node(grid, target):
    """Finds nearest free cell (0) to a target (row, col)."""
    if grid[int(target[0]), int(target[1])] == 0:
        return (int(target[0]), int(target[1]))

    free_cells = np.argwhere(grid == 0)
    if len(free_cells) == 0:
        raise ValueError("The grid has no free space available!")

    distances = np.linalg.norm(free_cells - np.array(target), axis=1)
    nearest_idx = np.argmin(distances)
    nearest_node = tuple(int(x) for x in free_cells[nearest_idx])
    return nearest_node

def main():
    print("==========================================================")
    print("             LUNAR ROVER AI SIMULATOR                     ")
    print("==========================================================\n")

    ensure_directories()

    # 1. Processing
    print("[STEP 1] Processing Terrain Image...")
    processor = TerrainProcessor(image_path="assets/terrain.jpg")
    processor.load_image()
    threshold_img = processor.process_terrain()

    # 2. Obstacle Detection
    print("[STEP 2] Running Obstacle Detection...")
    detector = ObstacleDetector(processor.original_image, threshold_img)
    annotated_img, count = detector.detect_obstacles(min_area=150)
    detector.save_output("outputs/obstacle_detection.png")

    # 3. Occupancy Grid
    print("[STEP 3] Generating Occupancy Grid...")
    grid_builder = OccupancyGrid(threshold_img, grid_size=(40, 40))
    grid = grid_builder.generate_grid(obstacle_ratio_thresh=0.08)
    grid_builder.save_grid_image("outputs/occupancy_grid.png")

    # 4. Start and Goal (Navigable ground region)
    requested_start = (35, 5)
    requested_goal = (18, 35)

    start_pos = find_nearest_free_node(grid, requested_start)
    goal_pos = find_nearest_free_node(grid, requested_goal)

    print(f"[STEP 4] Initializing A* Path Planner from {start_pos} to {goal_pos}...")
    
    orig_rgb = cv2.cvtColor(processor.original_image, cv2.COLOR_BGR2RGB)
    obs_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    print("[STEP 5] Launching Mission Animation Dashboard...")
    sim = RoverSimulation(
        original_img=orig_rgb,
        obstacle_img=obs_rgb,
        grid=grid,
        start=start_pos,
        goal=goal_pos
    )
    
    # Position dynamic obstacle in front of rover's path on open ground
    sim.dynamic_obstacle_pos = (28, 20)
    
    sim.run()

if __name__ == "__main__":
    main()