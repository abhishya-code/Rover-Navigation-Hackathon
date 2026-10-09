import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from astar import AStarPlanner

class RoverSimulation:
    def __init__(self, original_img, obstacle_img, grid, start, goal):
        self.original_img = original_img
        self.obstacle_img = obstacle_img
        self.grid = grid.copy()
        self.start = start
        self.goal = goal

        # --- AFTER ---
        self.planner = AStarPlanner(self.grid)
        self.path, self.explored = self.planner.plan(self.start, self.goal)

        if not self.path:
            print("[CRITICAL] Could not compute an initial path. Defaulting to start node.")
            self.path = [self.start]

        # Dynamic Obstacle Injection Parameters
        self.dynamic_obstacle_step = 8  # Step index when new obstacle appears
        self.dynamic_obstacle_pos = (20, 20)  # Mid-grid location
        self.replanned = False

    def run(self):
        """Creates 4 display subplots and animates the rover navigation."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.canvas.manager.set_window_title("LunarRover AI - Exploration Dashboard")

        ax_orig, ax_obs = axes[0, 0], axes[0, 1]
        ax_grid, ax_sim = axes[1, 0], axes[1, 1]

        # 1. Original Terrain View
        ax_orig.set_title("1. Original Terrain")
        ax_orig.imshow(self.original_img)
        ax_orig.axis("off")

        # 2. Obstacle Detection View
        ax_obs.set_title("2. Obstacle Detection")
        ax_obs.imshow(self.obstacle_img)
        ax_obs.axis("off")

        # 3. Occupancy Grid View
        ax_grid.set_title("3. Occupancy Grid")
        ax_grid.imshow(self.grid, cmap="gray_r")
        if self.explored:
            exp_r, exp_c = zip(*self.explored)
            ax_grid.scatter(exp_c, exp_r, c="cyan", s=5, alpha=0.3, label="Explored Nodes")
        ax_grid.plot(self.start[1], self.start[0], "go", markersize=8, label="Start")
        ax_grid.plot(self.goal[1], self.goal[0], "ro", markersize=8, label="Goal")
        ax_grid.legend(loc="upper right", fontsize=8)

        # 4. Rover Navigation Simulation View
        ax_sim.set_title("4. Rover Navigation Simulation")
        grid_plot = ax_sim.imshow(self.grid, cmap="binary")
        
        # Plot initial path
        path_r, path_c = zip(*self.path)
        (path_line,) = ax_sim.plot(path_c, path_r, "b-", linewidth=2, label="Planned Path")
        (rover_dot,) = ax_sim.plot([], [], "go", markersize=10, label="Rover Position")
        ax_sim.plot(self.start[1], self.start[0], "gs", markersize=8)
        ax_sim.plot(self.goal[1], self.goal[0], "rs", markersize=8)
        
        status_text = ax_sim.text(
            0.02, 0.95, "", transform=ax_sim.transAxes,
            color="yellow", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7)
        )

        legend_elements = [
            Patch(facecolor='green', edgecolor='g', label='Rover'),
            Patch(facecolor='red', edgecolor='r', label='Obstacle'),
            Patch(facecolor='blue', edgecolor='b', label='Planned Path')
        ]
        ax_sim.legend(handles=legend_elements, loc="upper right", fontsize=8)

        current_path = list(self.path)

        def update(frame):
            nonlocal current_path
            
            # Dynamic obstacle injection step
            if frame == self.dynamic_obstacle_step and not self.replanned:
                dyn_r, dyn_c = self.dynamic_obstacle_pos
                # Block a 3x3 block around target
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if 0 <= dyn_r+dr < self.grid.shape[0] and 0 <= dyn_c+dc < self.grid.shape[1]:
                            self.grid[dyn_r+dr, dyn_c+dc] = 1

                grid_plot.set_data(self.grid)
                current_rover_pos = current_path[min(frame, len(current_path)-1)]

                print(f"\n[ALERT] Dynamic obstacle detected at grid {self.dynamic_obstacle_pos}!")
                print("[INFO] Re-planning trajectory using A* Algorithm...")

                self.planner.grid = self.grid
                new_path, _ = self.planner.plan(current_rover_pos, self.goal)

                if new_path:
                    current_path = current_path[:frame] + new_path
                    self.replanned = True
                    print("[SUCCESS] New safe route found!\n")
                else:
                    print("[ERROR] Trajectory blocked completely!")

            idx = min(frame, len(current_path) - 1)
            curr_pos = current_path[idx]

            rover_dot.set_data([curr_pos[1]], [curr_pos[0]])

            cur_r, cur_c = zip(*current_path)
            path_line.set_data(cur_c, cur_r)

            if idx == len(current_path) - 1:
                status_text.set_text("MISSION SUCCESSFUL!")
                status_text.set_color("lime")
            elif self.replanned and frame >= self.dynamic_obstacle_step:
                status_text.set_text("PATH RE-PLANNED!")
                status_text.set_color("orange")
            else:
                status_text.set_text("NAVIGATING...")
                status_text.set_color("cyan")

            return rover_dot, path_line, grid_plot, status_text

        frames_total = len(self.path) + 15  # Buffer frames to allow viewing mission completion
        anim = FuncAnimation(
            fig, update, frames=frames_total, interval=250, repeat=False, blit=False
        )

        plt.tight_layout()
        
        # Save final path figure output
        fig.savefig("outputs/final_path.png", dpi=150)
        print("[INFO] Final navigation result saved to 'outputs/final_path.png'.")

        plt.show()