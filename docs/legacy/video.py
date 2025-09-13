import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

# -------------------------------------------------------
# exibicao_cronometrada.py (versão com gravação linke.mp4)
# -------------------------------------------------------

# ---- Constantes e Configurações ----
MODEL_NAME = r"DeepL\Controle_de_estacionamento\yolov8x.pt"  # Modelo pré-treinado
MODEL_CONFIDENCE_THRESHOLD = 0.1  # Limiar para detecção
WINDOW_NAME = "Estacionamento"

POLYGONS_FILE = r"DeepL\Controle_de_estacionamento\polygons.json"  # Nome do arquivo de persistência
VIDEO_FILE    = r"DeepL\Controle_de_estacionamento\parking.mp4"

# ---- Variáveis Globais para Polígonos ----
polygons = []
polygon_thresholds = []  # lista paralela a 'polygons'
current_polygon = []

# ---- Carregar Modelo YOLO ----
model = YOLO(MODEL_NAME)
CLASS_NAMES_DICT = model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    cid for cid, cname in CLASS_NAMES_DICT.items()
    if cname in SELECTED_CLASS_NAMES
]

# ----------------------------------------------------------------
# Funções de salvar/carregar polígonos
# ----------------------------------------------------------------
def load_polygons_from_file():
    """Carrega polígonos e thresholds salvos em um arquivo JSON (se existir)."""
    if os.path.exists(POLYGONS_FILE):
        with open(POLYGONS_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "polygons" in data and "thresholds" in data:
                loaded_polygons = [list(map(tuple, poly)) for poly in data["polygons"]]
                loaded_thresholds = data["thresholds"]
                return loaded_polygons, loaded_thresholds
            else:
                # Formato antigo (apenas polígonos); assume threshold=2
                loaded_polygons = [list(map(tuple, poly)) for poly in data]
                loaded_thresholds = [2] * len(loaded_polygons)
                return loaded_polygons, loaded_thresholds
    return [], []

def save_polygons_to_file(polygons_list, thresholds_list):
    """Salva polígonos e thresholds em formato JSON."""
    data = {
        "polygons": [list(map(list, poly)) for poly in polygons_list],
        "thresholds": thresholds_list
    }
    with open(POLYGONS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------------------------------------------
# Função de detecção de veículos com YOLO
# ----------------------------------------------------------------
def detect_vehicles(frame):
    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            if conf >= MODEL_CONFIDENCE_THRESHOLD and class_id in SELECTED_CLASS_IDS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'confidence': conf
                })
    return detections

# ----------------------------------------------------------------
# Funções utilitárias de desenho
# ----------------------------------------------------------------

def fade_value(t, in_start, in_end, out_start=None, out_end=None):
    """
    Retorna um valor de [0.0 a 1.0] dependendo de t:
      - Se t < in_start => 0
      - in_start <= t < in_end => interpola linearmente 0->1
      - in_end <= t < out_start (se existir) => 1
      - out_start <= t < out_end (se existir) => interpola linearmente 1->0
      - t >= out_end => 0
    """
    if t < in_start:
        return 0.0
    if t < in_end:
        # Subindo de 0 a 1
        return (t - in_start) / (in_end - in_start)
    # Já passou do in_end
    if (out_start is None) or (out_end is None):
        # Se não há fade_out definido, fica em 1 para sempre
        return 1.0
    # Se existe fade_out, mas ainda não chegou no out_start:
    if t < out_start:
        return 1.0
    if t < out_end:
        # Descendo de 1 a 0
        return 1.0 - (t - out_start) / (out_end - out_start)
    # Já passou do out_end
    return 0.0

def draw_overlay_with_alpha(base_img, overlay_func, alpha):
    """
    Cria uma imagem overlay chamando 'overlay_func(overlay)', 
    e depois faz addWeighted com alpha sobre base_img.
    Retorna a imagem final.
    """
    if alpha <= 0.0:
        # Nada para desenhar
        return base_img
    # Cria cópia para desenhar
    overlay = base_img.copy()
    # Chama a função de desenho que age sobre overlay
    overlay_func(overlay)
    # Combina
    return cv2.addWeighted(overlay, alpha, base_img, 1.0 - alpha, 0)

def put_text_with_shadow(img, text, org, font, font_scale, color, thickness=1):
    """Desenha texto com uma sombra preta para melhor leitura."""
    shadow_offset = (1, 1)
    cv2.putText(img, text,
                (org[0] + shadow_offset[0], org[1] + shadow_offset[1]),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

# ----------------------------------------------------------------
# Lógica principal
# ----------------------------------------------------------------
def main():
    global polygons, polygon_thresholds

    # Carrega polígonos
    polygons, polygon_thresholds = load_polygons_from_file()
    print(f"Carregado {len(polygons)} polígonos, thresholds = {polygon_thresholds}")

    # Abre vídeo
    video = cv2.VideoCapture(VIDEO_FILE)
    if not video.isOpened():
        print("Não foi possível abrir o vídeo.")
        return

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0
    print(f"FPS detectado: {fps:.2f}. Duração aproximada do vídeo: {duration:.1f} seg.")

    # Prepara VideoWriter para "linke.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("linke.mp4", fourcc, fps, (width, height))

    # ---------------------------
    # Prepara estado das vagas
    # ---------------------------
    num_slots = len(polygons)
    occupant_state = ["free"] * num_slots
    occupant_change_timestamp = [0.0] * num_slots

    def update_slot_state(i, is_detected, current_time):
        old_state = occupant_state[i]
        if old_state == "free":
            if is_detected:
                occupant_state[i] = "pending_occupied"
                occupant_change_timestamp[i] = current_time
        elif old_state == "pending_occupied":
            if is_detected:
                if (current_time - occupant_change_timestamp[i]) >= 3.0:
                    occupant_state[i] = "occupied"
                    occupant_change_timestamp[i] = current_time
            else:
                occupant_state[i] = "free"
        elif old_state == "occupied":
            if not is_detected:
                occupant_state[i] = "pending_free"
                occupant_change_timestamp[i] = current_time
        elif old_state == "pending_free":
            if not is_detected:
                if (current_time - occupant_change_timestamp[i]) >= 8.0:
                    occupant_state[i] = "free"
                    occupant_change_timestamp[i] = current_time
            else:
                occupant_state[i] = "occupied"

    first_frame = True
    frame_index = 0

    start_time = time.time()  # Para nosso controle de delta real (mas poderia usar time do sistema mesmo)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Tempo (em segundos) relativo ao início do vídeo
        t = frame_index / fps

        # Detecta veículos
        detections = detect_vehicles(frame)

        # Descobre quais vagas foram "atingidas" neste frame
        detected_slots_this_frame = set()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            margin_x = int(0.2 * (x2 - x1))
            margin_y = int(0.2 * (y2 - y1))

            # Alguns pontos dentro da bbox
            points = [
                (x1 + margin_x, y1 + margin_y),
                (x2 - margin_x, y1 + margin_y),
                ((x1 + x2)//2, (y1 + y2)//2),
                (x1 + margin_x, y2 - margin_y),
                (x2 - margin_x, y2 - margin_y)
            ]

            slot_index = -1
            for i, poly in enumerate(polygons):
                poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                inside_points = sum(
                    1 for cp in points if cv2.pointPolygonTest(poly_np, cp, False) >= 0
                )
                if inside_points >= polygon_thresholds[i]:
                    slot_index = i
                    break

            if slot_index != -1:
                detected_slots_this_frame.add(slot_index)

        # Atualiza estado das vagas
        current_time = time.time()
        for i in range(num_slots):
            update_slot_state(i, (i in detected_slots_this_frame), current_time)

        # Se for o primeiro frame, as vagas detectadas são ocupadas imediatamente
        if first_frame:
            for i in range(num_slots):
                if i in detected_slots_this_frame:
                    occupant_state[i] = "occupied"
                    occupant_change_timestamp[i] = current_time
            first_frame = False

        # --------------------------------------------------------------
        # Construímos a imagem final passo a passo, usando camadas
        # --------------------------------------------------------------
        frame_display = frame.copy()

        # 1) Desenho dos bounding boxes + pontos
        def draw_bounding_boxes(overlay_img):
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_id = det['class_id']
                conf = det['confidence']

                # bounding box
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (128, 0, 128), 2)

                label = f"{CLASS_NAMES_DICT[class_id]} {conf*100:.1f}%"
                cv2.putText(overlay_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # desenha os pontos
                margin_x = int(0.2 * (x2 - x1))
                margin_y = int(0.2 * (y2 - y1))
                points = [
                    (x1 + margin_x, y1 + margin_y),
                    (x2 - margin_x, y1 + margin_y),
                    ((x1 + x2)//2, (y1 + y2)//2),
                    (x1 + margin_x, y2 - margin_y),
                    (x2 - margin_x, y2 - margin_y)
                ]
                for p in points:
                    cv2.circle(overlay_img, p, 5, (0, 255, 0), -1)

        # Faz fade in/out do bounding box
        bbox_alpha = fade_value(t, 5, 6, 30, 31)
        frame_display = draw_overlay_with_alpha(frame_display, draw_bounding_boxes, bbox_alpha)

        # 2) Desenho dos polígonos (ocupado/livre)
        def draw_polygons(overlay_img):
            for i, poly in enumerate(polygons):
                poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                state = occupant_state[i]
                # Considera 'occupied' ou 'pending_free' como vermelho
                is_occupied = (state in ("occupied", "pending_free"))
                color = (0, 255, 0) if not is_occupied else (0, 0, 255)
                cv2.polylines(overlay_img, [poly_np], True, color, 2)

        poly_alpha = fade_value(t, 10, 11, 53, 54)
        frame_display = draw_overlay_with_alpha(frame_display, draw_polygons, poly_alpha)

        # 3) Desenho do painel de informações
        def draw_info(overlay_img):
            vagas_ocupadas = 0
            for i in range(num_slots):
                if occupant_state[i] in ("occupied", "pending_free"):
                    vagas_ocupadas += 1
            vagas_livres = num_slots - vagas_ocupadas

            # Painel semi-transparente
            overlay = overlay_img.copy()
            panel_x, panel_y = 10, 10
            panel_w, panel_h = 370, 220
            cv2.rectangle(overlay, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h),
                          (30, 30, 30), -1)
            alpha_panel = 0.6
            cv2.addWeighted(overlay, alpha_panel, overlay_img, 1 - alpha_panel, 0, overlay_img)

            # Textos
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_spacing = 35
            put_text_with_shadow(overlay_img,
                                 f"Total de vagas: {num_slots}",
                                 (panel_x + 10, panel_y + 30),
                                 font, 0.7, (255, 255, 255), 2)
            put_text_with_shadow(overlay_img,
                                 f"Ocupadas: {vagas_ocupadas}",
                                 (panel_x + 10, panel_y + 30 + line_spacing),
                                 font, 0.7, (0, 0, 255), 2)
            put_text_with_shadow(overlay_img,
                                 f"Livres: {vagas_livres}",
                                 (panel_x + 10, panel_y + 30 + 2*line_spacing),
                                 font, 0.7, (0, 255, 0), 2)

            put_text_with_shadow(overlay_img,
                                 "Porcentagem de vagas ocupadas:",
                                 (panel_x + 10, panel_y + 30 + 3*line_spacing),
                                 font, 0.6, (0, 255, 255), 2)

            # Barra de ocupação
            bar_x = panel_x + 10
            bar_y = panel_y + 30 + 4*line_spacing
            bar_w, bar_h = 250, 25
            cv2.rectangle(overlay_img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                          (255,255,255), 2)
            occupancy_ratio = (vagas_ocupadas / num_slots) if num_slots > 0 else 0
            fill_w = int(bar_w * occupancy_ratio)
            cv2.rectangle(overlay_img, (bar_x, bar_y),
                          (bar_x+fill_w, bar_y+bar_h), (0,0,255), -1)

            pct_text = f"{int(occupancy_ratio*100)}%"
            put_text_with_shadow(overlay_img, pct_text,
                                 (bar_x + bar_w + 15, bar_y + bar_h),
                                 font, 0.7, (255, 255, 255), 2)

        info_alpha = fade_value(t, 15, 16)
        frame_display = draw_overlay_with_alpha(frame_display, draw_info, info_alpha)

        # Escreve frame no vídeo de saída
        out.write(frame_display)

        # Exibe na tela
        cv2.imshow(WINDOW_NAME, frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        frame_index += 1

    # Libera
    video.release()
    out.release()
    cv2.destroyAllWindows()
    print("Vídeo 'linke.mp4' gerado com sucesso.")

if __name__ == "__main__":
    main()