import os
import cv2
import numpy as np

def ensure_directories():
    """Ensure assets/ and outputs/ directories exist."""
    os.makedirs("assets", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def generate_synthetic_terrain(file_path="assets/terrain.jpg", width=400, height=400):
    """
    Generates a synthetic lunar terrain image with rocks and craters 
    if no image is provided.
    """
    ensure_directories()
    
    # Create light gray background representing lunar dust/regolith
    terrain = np.full((height, width, 3), 180, dtype=np.uint8)
    
    # Add random background noise/texture
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    terrain = np.clip(terrain.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw dark obstacles (Rocks and Craters)
    # Circle 1 (Large Crater)
    cv2.circle(terrain, (100, 100), 35, (40, 40, 40), -1)
    # Circle 2 (Medium Rock Cluster)
    cv2.circle(terrain, (250, 120), 45, (30, 30, 30), -1)
    # Rectangle/Boulder 1
    cv2.rectangle(terrain, (150, 220), (220, 290), (20, 20, 20), -1)
    # Circle 3 (Small Obstacle)
    cv2.circle(terrain, (300, 300), 30, (50, 50, 50), -1)
    # Irregular Rock
    pts = np.array([[50, 280], [90, 260], [120, 310], [70, 340]], np.int32)
    cv2.fillPoly(terrain, [pts], (35, 35, 35))

    cv2.imwrite(file_path, terrain)
    print(f"[INFO] Synthetic terrain generated and saved to '{file_path}'.")