import json
import os
from typing import List, Tuple

import cv2
import numpy as np


def load_polygons(path: str) -> Tuple[List[List[Tuple[int, int]]], List[int]]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "polygons" in data and "thresholds" in data:
                loaded_polygons = [list(map(tuple, poly)) for poly in data["polygons"]]
                loaded_thresholds = data["thresholds"]
                return loaded_polygons, loaded_thresholds
            else:
                loaded_polygons = [list(map(tuple, poly)) for poly in data]
                loaded_thresholds = [2] * len(loaded_polygons)
                return loaded_polygons, loaded_thresholds
    return [], []


def save_polygons(path: str, polygons: List[List[Tuple[int, int]]], thresholds: List[int]) -> None:
    data = {
        "polygons": [list(map(list, poly)) for poly in polygons],
        "thresholds": thresholds,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def draw_polygons_interactive(frame,
                              window_name: str,
                              polygons: List[List[Tuple[int, int]]],
                              thresholds: List[int],
                              save_path: str) -> List[List[Tuple[int, int]]]:
    current_polygon: List[Tuple[int, int]] = []

    base_frame = frame.copy()
    h, w = base_frame.shape[:2]

    # Define limites de exibição para caber em telas comuns (evita corte)
    max_w, max_h = 1280, 720
    scale_x = min(1.0, max_w / float(w))
    scale_y = min(1.0, max_h / float(h))
    scale = min(scale_x, scale_y)
    disp_w, disp_h = int(w * scale), int(h * scale)

    def to_disp(pt: Tuple[int, int]) -> Tuple[int, int]:
        if scale == 1.0:
            return pt
        return int(pt[0] * scale), int(pt[1] * scale)

    def to_orig(pt: Tuple[int, int]) -> Tuple[int, int]:
        if scale == 1.0:
            return pt
        return int(round(pt[0] / scale)), int(round(pt[1] / scale))

    def draw_all_polys() -> np.ndarray:
        # Desenha convertendo pontos originais para a escala de exibição
        if scale == 1.0:
            output_img = base_frame.copy()
        else:
            output_img = cv2.resize(base_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        for poly in polygons:
            if len(poly) > 0:
                scaled_poly = [to_disp(p) for p in poly]
                for i in range(len(scaled_poly) - 1):
                    cv2.line(output_img, scaled_poly[i], scaled_poly[i + 1], (144, 214, 79), 1)
                if len(scaled_poly) > 2:
                    cv2.line(output_img, scaled_poly[-1], scaled_poly[0], (144, 214, 79), 1)
                for (px, py) in scaled_poly:
                    cv2.circle(output_img, (px, py), 3, (144, 214, 79), -1)

        if len(current_polygon) > 0:
            scaled_current = [to_disp(p) for p in current_polygon]
            for i in range(len(scaled_current) - 1):
                cv2.line(output_img, scaled_current[i], scaled_current[i + 1], (0, 255, 0), 1)
            for (cx, cy) in scaled_current:
                cv2.circle(output_img, (cx, cy), 3, (0, 255, 0), -1)

        # Desenha o painel de ajuda com as teclas
        def draw_help(overlay_img: np.ndarray) -> None:
            panel_x, panel_y = 10, 10
            panel_w, panel_h = 410, 180
            overlay = overlay_img.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, overlay_img, 0.4, 0, overlay_img)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.5
            lh = 20
            x = panel_x + 10
            y = panel_y + 22

            cv2.putText(overlay_img, 'Ajuda (desenho de vagas):', (x, y), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, '- Clique: adiciona ponto', (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, "- f: finaliza o poligono (nao salva)", (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, "- Espaco: finaliza e salva o poligono", (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, '- c: remove o ultimo poligono salvo', (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, '- Backspace: desfaz o ultimo ponto', (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, '- 1: marca ultimo poligono (threshold=1)', (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)
            y += lh
            cv2.putText(overlay_img, '- q/ESC: sair', (x, y), font, fs, (200, 255, 200), 1, cv2.LINE_AA)

        draw_help(output_img)

        return output_img

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append(to_orig((x, y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_img = draw_all_polys()
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                thresholds.append(2)
                print("Polígono finalizado (não salvo). Threshold=2.")
            else:
                print("Polígono inválido, descartado.")
            current_polygon.clear()

        elif key == ord(' '):
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                thresholds.append(2)
                save_polygons(save_path, polygons, thresholds)
                print("Polígono salvo! Threshold=2.")
            else:
                print("Polígono inválido, descartado.")
            current_polygon.clear()

        elif key == ord('c'):
            if polygons:
                polygons.pop()
                thresholds.pop()
                save_polygons(save_path, polygons, thresholds)
                print("Último polígono removido e arquivo atualizado.")
            else:
                print("Não há polígonos para remover.")

        elif key == 8:  # Backspace
            if current_polygon:
                current_polygon.pop()
            else:
                print("Nenhum ponto no polígono atual.")

        elif key == ord('1'):
            if polygons:
                thresholds[-1] = 1
                save_polygons(save_path, polygons, thresholds)
                print("Threshold do último polígono alterado para 1 e salvo.")
            else:
                print("Nenhum polígono para alterar.")

        elif key in [ord('q'), 27]:
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                thresholds.append(2)
            break

    cv2.destroyWindow(window_name)
    return polygons


def bbox_checkpoints(x1: int, y1: int, x2: int, y2: int):
    margin_x = int(0.2 * (x2 - x1))
    margin_y = int(0.2 * (y2 - y1))
    return [
        (x1 + margin_x, y1 + margin_y),
        (x2 - margin_x, y1 + margin_y),
        ((x1 + x2) // 2, (y1 + y2) // 2),
        (x1 + margin_x, y2 - margin_y),
        (x2 - margin_x, y2 - margin_y),
    ]


def first_polygon_satisfying(points, polygons: List[List[Tuple[int, int]]], thresholds: List[int]) -> int:
    for i, poly in enumerate(polygons):
        poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        inside_points = sum(1 for p in points if cv2.pointPolygonTest(poly_np, p, False) >= 0)
        if inside_points >= thresholds[i]:
            return i
    return -1
