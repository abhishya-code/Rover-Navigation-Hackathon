import cv2
import numpy as np

class OccupancyGrid:
    def __init__(self, threshold_image, grid_size=(40, 40)):
        self.threshold_image = threshold_image
        self.grid_size = grid_size
        self.grid = None

    def generate_grid(self, obstacle_ratio_thresh=0.08):
        """
        Converts threshold image into binary matrix.
        0 = Free space (White on display)
        1 = Obstacle (Black on display)
        """
        h, w = self.threshold_image.shape
        grid_rows, grid_cols = self.grid_size
        cell_h = h / grid_rows
        cell_w = w / grid_cols

        # Default all to free space (0)
        self.grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

        # Mark sky (top 25%) as Obstacle/Out-of-Bounds (1)
        sky_rows = int(grid_rows * 0.25)
        self.grid[0:sky_rows, :] = 1

        for r in range(sky_rows, grid_rows):
            for c in range(grid_cols):
                r_start, r_end = int(r * cell_h), int((r + 1) * cell_h)
                c_start, c_end = int(c * cell_w), int((c + 1) * cell_w)

                cell_crop = self.threshold_image[r_start:r_end, c_start:c_end]
                obstacle_ratio = np.count_nonzero(cell_crop) / float(cell_crop.size)

                if obstacle_ratio > obstacle_ratio_thresh:
                    self.grid[r, c] = 1  # Obstacle

        return self.grid

    def save_grid_image(self, output_path="outputs/occupancy_grid.png"):
        grid_visual = np.where(self.grid == 1, 0, 255).astype(np.uint8)
        grid_resized = cv2.resize(grid_visual, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(output_path, grid_resized)
        print(f"[INFO] Saved occupancy grid to '{output_path}'.")