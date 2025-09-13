from typing import List, Tuple
import cv2
import numpy as np


def put_text_with_shadow(img, text, org, font, font_scale, color, thickness=1):
    shadow_offset = (1, 1)
    cv2.putText(img, text,
                (org[0] + shadow_offset[0], org[1] + shadow_offset[1]),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_polylines_by_state(img, polygons: List[List[Tuple[int, int]]], states: List[str]) -> None:
    for i, poly in enumerate(polygons):
        poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        is_occupied = states[i] in ("occupied", "pending_free")
        color = (0, 255, 0) if not is_occupied else (0, 0, 255)
        cv2.polylines(img, [poly_np], isClosed=True, color=color, thickness=2)


def draw_info_panel(img, total_slots: int, occupied: int) -> None:
    free = total_slots - occupied
    overlay = img.copy()
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 370, 220
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    alpha_panel = 0.6
    cv2.addWeighted(overlay, alpha_panel, img, 1 - alpha_panel, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_spacing = 35
    put_text_with_shadow(img, f"Total de vagas: {total_slots}", (panel_x + 10, panel_y + 30), font, 0.7, (255, 255, 255), 2)
    put_text_with_shadow(img, f"Ocupadas: {occupied}", (panel_x + 10, panel_y + 30 + line_spacing), font, 0.7, (0, 0, 255), 2)
    put_text_with_shadow(img, f"Livres: {free}", (panel_x + 10, panel_y + 30 + 2 * line_spacing), font, 0.7, (0, 255, 0), 2)
    put_text_with_shadow(img, "Porcentagem de vagas ocupadas:", (panel_x + 10, panel_y + 30 + 3 * line_spacing), font, 0.6, (0, 255, 255), 2)

    bar_x = panel_x + 10
    bar_y = panel_y + 30 + 4 * line_spacing
    bar_w, bar_h = 250, 25
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    occupancy_ratio = (occupied / total_slots) if total_slots > 0 else 0.0
    fill_w = int(bar_w * occupancy_ratio)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)
    pct_text = f"{int(occupancy_ratio * 100)}%"
    put_text_with_shadow(img, pct_text, (bar_x + bar_w + 15, bar_y + bar_h), font, 0.7, (255, 255, 255), 2)


def draw_detection_boxes(img, detections: List[dict], class_names: dict | None = None) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det.get("class_id")
        conf = det.get("confidence")
        color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if cls_id is not None and conf is not None and class_names is not None:
            label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        elif cls_id is not None and class_names is not None:
            label = f"{class_names.get(cls_id, str(cls_id))}"
        elif conf is not None:
            label = f"{conf:.2f}"
        else:
            label = None

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
