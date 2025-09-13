from typing import List, Dict, Tuple
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = 0.2):
        self.model = YOLO(weights_path)
        self.class_names = self.model.names
        selected_classes = ["car", "motorcycle", "bus", "truck"]
        self.selected_ids = [cid for cid, cname in self.class_names.items() if cname in selected_classes]
        self.confidence_threshold = confidence_threshold

    def detect(self, frame) -> List[Dict[str, Tuple[int, int, int, int]]]:
        results = self.model(frame)
        detections: List[Dict] = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                if conf >= self.confidence_threshold and class_id in self.selected_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": conf,
                    })
        return detections
