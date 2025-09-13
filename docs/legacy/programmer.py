import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

# ---- Constantes e Configurações ----
MODEL_NAME = r"DeepL\Controle_de_estacionamento\yolov8x.pt"  # Modelo pré-treinado
MODEL_CONFIDENCE_THRESHOLD = 0.1  # Limiar para detecção
WINDOW_NAME = "Estacionamento"
POLYGONS_FILE = r"DeepL\Controle_de_estacionamento\polygons.json"  # Nome do arquivo de persistência

# ---- Variáveis Globais para Desenho de Polígonos ----
polygons = []
polygon_thresholds = []  # lista paralela a 'polygons' indicando quantos pontos mínimos cada polígono requer
current_polygon = []
img = None

# ---- Carregar Modelo YOLO pré-treinado COCO ----
model = YOLO(MODEL_NAME)

# Obtenção e filtragem das classes de interesse (veículos)
CLASS_NAMES_DICT = model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    class_id for class_id, class_name in CLASS_NAMES_DICT.items() 
    if class_name in SELECTED_CLASS_NAMES
]

# ---- Funções para carregar os polígonos ----
def load_polygons_from_file():
    """Carrega polígonos e thresholds salvos em um arquivo JSON (se existir). 
       Se o arquivo estiver no formato antigo (lista de listas), assume threshold=2 para todos.
    """
    if os.path.exists(POLYGONS_FILE):
        with open(POLYGONS_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "polygons" in data and "thresholds" in data:
                # Formato novo
                loaded_polygons = [list(map(tuple, poly)) for poly in data["polygons"]]
                loaded_thresholds = data["thresholds"]
                return loaded_polygons, loaded_thresholds
            else:
                # Formato antigo (apenas polígonos, sempre usando 2 pontos)
                loaded_polygons = [list(map(tuple, poly)) for poly in data]
                loaded_thresholds = [2] * len(loaded_polygons)
                return loaded_polygons, loaded_thresholds
    return [], []

# ---- Funções para salvar os polígonos ----
def save_polygons_to_file(polygons_list, thresholds_list):
    """Salva a lista de polígonos e a lista de thresholds em um arquivo JSON no formato novo:
       {
         "polygons": [...],
         "thresholds": [...]
       }
    """
    data = {
        "polygons": [list(map(list, poly)) for poly in polygons_list],
        "thresholds": thresholds_list
    }
    with open(POLYGONS_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ---- Desenha os polígonos finalizados e os em andamento sobre uma imagem base fornecida ----
def draw_all_polygons(base_img):
    """
    Desenha todos os polígonos finalizados (polygons) e
    o polígono em andamento (current_polygon) sobre a imagem base.
    """
    output_img = base_img.copy()
    
    # Desenha polígonos finalizados
    for poly in polygons:
        if len(poly) > 1:
            for i in range(len(poly) - 1):
                cv2.line(output_img, poly[i], poly[i+1], (144, 214, 79), 1)
            # Fecha o polígono
            cv2.line(output_img, poly[-1], poly[0], (144, 214, 79), 1)
        
        # (Opcional) desenha pequenos círculos nos vértices:
        for (px, py) in poly:
            cv2.circle(output_img, (px, py), 3, (144, 214, 79), -1)

    # Desenha o polígono atual
    if len(current_polygon) > 1:
        for i in range(len(current_polygon) - 1):
            cv2.line(output_img, current_polygon[i], current_polygon[i+1], (0, 255, 0), 1)
    # (Opcional) desenha pequenos círculos nos vértices do polígono atual
    for (cx, cy) in current_polygon:
        cv2.circle(output_img, (cx, cy), 3, (0, 255, 0), -1)
    
    return output_img

# ---- Interface interativa para criar, editar e salvar polígonos sobre uma imagem ----
def draw_polygons_and_get_points(frame):
    """
    Permite ao usuário desenhar polígonos na imagem dada (frame).
    - 'f' finaliza o polígono atual (não salva em disco).
    - ' ' (barra de espaço) finaliza e salva o polígono atual.
    - 'c' remove o último polígono salvo.
    - Backspace remove o último ponto do polígono atual.
    - 'q' ou 'Esc' finaliza o programa, salvando o polígono atual se tiver >=3 pontos.
    - (NOVO) '1' faz com que o último polígono finalizado seja marcado com threshold = 1.
    """
    global img, polygons, polygon_thresholds, current_polygon
    
    base_frame = frame.copy()
    img = base_frame.copy()
    cv2.namedWindow(WINDOW_NAME)

    def mouse_callback(event, x, y, flags, param):
        global current_polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
    
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        temp_img = draw_all_polygons(base_frame)
        cv2.imshow(WINDOW_NAME, temp_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('f'):
            # Finaliza o polígono atual, MAS NÃO SALVA
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                polygon_thresholds.append(2)  # Por padrão, 2 pontos
                print("Polígono finalizado (não salvo em disco). Threshold=2.")
            else:
                print("Polígono inválido (menos de 3 pontos). Descartado.")
            current_polygon.clear()

        elif key == ord(' '):
            # Finaliza o polígono atual E SALVA em disco
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                polygon_thresholds.append(2)  # Por padrão, 2 pontos
                save_polygons_to_file(polygons, polygon_thresholds)
                print("Polígono salvo em disco! Threshold=2.")
            else:
                print("Polígono inválido (menos de 3 pontos). Descartado.")
            current_polygon.clear()

        elif key == ord('c'):
            # Remove o último polígono SALVO (se existir) e atualiza o arquivo
            if polygons:
                polygons.pop()
                polygon_thresholds.pop() 
                save_polygons_to_file(polygons, polygon_thresholds)
                print("Último polígono removido e arquivo atualizado.")
            else:
                print("Não há polígonos para remover.")

        elif key == 8:  # Backspace
            # Desfazer último ponto do polígono em andamento
            if len(current_polygon) > 0:
                current_polygon.pop()
            else:
                print("Não há pontos para remover no polígono atual.")

        elif key == ord('1'):
            # Ajusta threshold para 1 ponto na última vaga finalizada
            if polygons:
                polygon_thresholds[-1] = 1
                save_polygons_to_file(polygons, polygon_thresholds)
                print("Threshold do último polígono alterado para 1 e salvo.")
            else:
                print("Não há polígonos finalizados para alterar threshold.")

        elif key == ord('q') or key == 27:  # 'q' ou ESC
            if len(current_polygon) >= 3:
                polygons.append(current_polygon.copy())
                polygon_thresholds.append(2)
            break

    cv2.destroyWindow(WINDOW_NAME)
    return polygons

# ---- Função de Detecção de Veículos utilizando YOLO ----
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

def main():
    global polygons, polygon_thresholds

    # Carrega os polígonos existentes
    polygons, polygon_thresholds = load_polygons_from_file()
    print(f"Polígonos carregados do arquivo: {len(polygons)} encontrados.")
    print(f"Thresholds carregados: {polygon_thresholds}")

    video = cv2.VideoCapture(r'DeepL\Controle_de_estacionamento\parking.mp4')
    if not video.isOpened():
        print("Não foi possível abrir o vídeo.")
        return

    # Avança para 57s do vídeo (opcional, conforme sua lógica original)
    video.set(cv2.CAP_PROP_POS_MSEC, 57000)
    ret, frame = video.read()
    if not ret:
        print("Não foi possível capturar o frame aos 57s.")
        return

    # Permite desenhar (e salvar) polígonos no frame capturado
    draw_polygons_and_get_points(frame)
    # Se ao final, não houver nenhum polígono (nem carregado, nem desenhado), saímos
    if len(polygons) == 0:
        print("Nenhum polígono disponível.")
        return

    # Reinicia o vídeo
    video.set(cv2.CAP_PROP_POS_MSEC, 0)
    cv2.namedWindow(WINDOW_NAME)

    # -----------------------------------------------------------------------------
    # NOVA LÓGICA DE MÁQUINA DE ESTADOS PARA CONTROLAR O ATRASO DE 3s
    # -----------------------------------------------------------------------------

    # Estados possíveis para cada vaga:
    #   "free"             = vaga livre confirmada
    #   "pending_occupied" = detectamos veículo, mas ainda esperando os 3s para confirmar ocupada
    #   "occupied"         = vaga ocupada confirmada
    #   "pending_free"     = detectamos que saiu o veículo, mas aguardando 3s para confirmar livre
    #
    # occupant_state[i] = estado atual da vaga i
    # occupant_change_timestamp[i] = armazena quando iniciamos a mudança de estado
    #
    # OBS: Vagas que já começam com veículo devem ser consideradas ocupadas imediatamente.
    #      Faremos isso logo após a primeira análise de frame.
    # -----------------------------------------------------------------------------

    num_slots = len(polygons)
    occupant_state = ["free"] * num_slots
    occupant_change_timestamp = [0.0] * num_slots

    # Função utilitária para atualizar o estado de uma vaga com base em se há detecção ou não
    def update_slot_state(i, is_detected, current_time):
        old_state = occupant_state[i]

        if old_state == "free":
            if is_detected:
                # Inicia contagem para ficar ocupada
                occupant_state[i] = "pending_occupied"
                occupant_change_timestamp[i] = current_time

        elif old_state == "pending_occupied":
            if is_detected:
                # Continua detectado, checa se já passou 3s
                if (current_time - occupant_change_timestamp[i]) >= 3.0:
                    occupant_state[i] = "occupied"
                    occupant_change_timestamp[i] = current_time
            else:
                # Perdeu a detecção, volta a ficar livre
                occupant_state[i] = "free"

        elif old_state == "occupied":
            if not is_detected:
                # Inicia contagem para ficar livre
                occupant_state[i] = "pending_free"
                occupant_change_timestamp[i] = current_time

        elif old_state == "pending_free":
            if not is_detected:
                # Continua sem detecção, checa se já passou 3s
                if (current_time - occupant_change_timestamp[i]) >= 5.5:
                    occupant_state[i] = "free"
                    occupant_change_timestamp[i] = current_time
            else:
                # Voltou a detectar veículo
                occupant_state[i] = "occupied"

    # Vamos processar quadro a quadro
    first_frame = True
    while True:
        ret, frame = video.read()
        if not ret:
            break

        current_time = time.time()
        frame_display = frame.copy()
        detections = detect_vehicles(frame)

        # Descobrindo qual vaga foi detectada neste frame
        # (cada det só pode "contar" pra primeira vaga que satisfizer o threshold)
        detected_slots_this_frame = set()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Pequena margem
            margin_x = int(0.2 * (x2 - x1))
            margin_y = int(0.2 * (y2 - y1))

            points = [
                (x1 + margin_x, y1 + margin_y),
                (x2 - margin_x, y1 + margin_y),
                ((x1 + x2)//2, (y1 + y2)//2),
                (x1 + margin_x, y2 - margin_y),
                (x2 - margin_x, y2 - margin_y)
            ]

            # Verifica o primeiro polígono que satisfizer o threshold
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
                # Desenho da bounding box e pontos só para efeito visual
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (128, 0, 128), 2)
                class_id = det['class_id']
                confidence = det['confidence']
                label = f"{CLASS_NAMES_DICT[class_id]} {confidence*100:.2f}%"
                cv2.putText(frame_display, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                for p in points:
                    cv2.circle(frame_display, p, 5, (0, 255, 0), -1)

        # ---------------------- Atualiza estados das vagas ----------------------
        for i in range(num_slots):
            is_detected = (i in detected_slots_this_frame)
            update_slot_state(i, is_detected, current_time)

        # Se for o primeiro frame, podemos forçar as que estão detectadas a ficarem "occupied" imediatamente,
        # para atender à exigência: "vagas que ao iniciarem o vídeo já estiverem com veículo, 
        # devem ser detectadas imediatamente".
        if first_frame:
            for i in range(num_slots):
                if i in detected_slots_this_frame:
                    occupant_state[i] = "occupied"
                    occupant_change_timestamp[i] = current_time
            first_frame = False

        # ----------------- Desenhar polígonos e contar ocupações -----------------
        vagas_ocupadas = 0
        for i, poly in enumerate(polygons):
            poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

            # Consideramos a vaga "ocupada" se o estado for "occupied" OU "pending_free"
            # ("pending_free" = ainda não liberou totalmente).
            state = occupant_state[i]
            if state in ("occupied", "pending_free"):
                is_occupied = True
            else:
                is_occupied = False

            color = (0, 255, 0) if not is_occupied else (0, 0, 255)
            cv2.polylines(frame_display, [poly_np], isClosed=True, color=color, thickness=2)

            # (Opcional) escrever o estado ou o threshold
            px, py = poly[0]
            cv2.putText(frame_display, f"{state} (th={polygon_thresholds[i]})", 
                        (px, py), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255,255,255), 1)

            if is_occupied:
                vagas_ocupadas += 1

        num_vagas = num_slots
        vagas_livres = num_vagas - vagas_ocupadas

        #info_text = (f"Numero de vagas: {num_vagas}  "
        #             f"Vagas ocupadas: {vagas_ocupadas}  "
        #             f"Vagas livres: {vagas_livres}")
        #cv2.putText(frame_display, info_text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow(WINDOW_NAME, frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para sair
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()