import cv2

class ObstacleDetector:
    def __init__(self, original_image, threshold_image):
        self.original_image = original_image.copy()
        self.threshold_image = threshold_image
        self.annotated_image = original_image.copy()
        self.obstacles_count = 0
        self.contours = []

    def detect_obstacles(self, min_area=150, max_area=10000):
        """Detects rock contours and marks them with bounding boxes."""
        contours, _ = cv2.findContours(
            self.threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.obstacles_count = 0
        self.contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                self.contours.append(cnt)
                self.obstacles_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Draw red box
                cv2.rectangle(self.annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        print(f"[INFO] Total Hazards Detected: {self.obstacles_count}")
        return self.annotated_image, self.obstacles_count

    def save_output(self, output_path="outputs/obstacle_detection.png"):
        cv2.imwrite(output_path, self.annotated_image)
        print(f"[INFO] Saved obstacle detection output to '{output_path}'.")