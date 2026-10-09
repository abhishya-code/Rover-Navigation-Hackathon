import os
import cv2
import numpy as np

class TerrainProcessor:
    def __init__(self, image_path="assets/terrain.jpg"):
        self.image_path = image_path
        self.original_image = None
        self.gray_image = None
        self.blurred_image = None
        self.threshold_image = None

    def load_image(self):
        """Loads terrain image."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"[ERROR] Could not read '{self.image_path}'. Check file path.")
        return self.original_image

    def process_terrain(self, blur_kernel=(9, 9), thresh_val=100):
        if self.original_image is None:
            self.load_image()

        h, w, _ = self.original_image.shape

        # 1. Grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Smooth out fine sand grains
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (11, 11), 0)

        # 3. Local Difference / High-pass filter to catch rocks (contrasting relative to ground)
        # Morphological Black Hat extracts dark rocks/shadows on lighter dirt
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(self.blurred_image, cv2.MORPH_BLACKHAT, kernel)
        
        # Top Hat extracts bright rock faces
        tophat = cv2.morphologyEx(self.blurred_image, cv2.MORPH_TOPHAT, kernel)
        
        # Combine both contrast features
        rock_features = cv2.add(blackhat, tophat)

        # 4. Binary Threshold on local feature contrast
        _, self.threshold_image = cv2.threshold(rock_features, 25, 255, cv2.THRESH_BINARY)

        # 5. Mask Sky (Top 25% of image)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask[0:int(h * 0.25), :] = 0  
        self.threshold_image = cv2.bitwise_and(self.threshold_image, mask)

        # 6. Clean specks with Morphological Close
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.threshold_image = cv2.morphologyEx(self.threshold_image, cv2.MORPH_CLOSE, close_kernel)

        return self.threshold_image