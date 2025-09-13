import argparse
import sys
import time
import yaml
import cv2

from .detector import VehicleDetector
from .polygons import load_polygons, save_polygons, draw_polygons_interactive, bbox_checkpoints, first_polygon_satisfying
from .occupancy import OccupancyStateMachine
from .visualization import draw_polylines_by_state, draw_info_panel
from .visualization import draw_detection_boxes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parking-monitor", description="Monitoramento de estacionamento com YOLOv8")
    p.add_argument("command", choices=["draw", "watch", "render"], help="Ação a executar")
    p.add_argument("--config", default="configs/config.yaml", help="Caminho do config.yaml")
    p.add_argument("--video", dest="video", help="Caminho do vídeo (sobrepõe config)")
    p.add_argument("--weights", dest="weights", help="Caminho dos pesos YOLO (sobrepõe config)")
    p.add_argument("--polygons", dest="polygons", help="Caminho do polygons.json (sobrepõe config)")
    p.add_argument("--draw-sec", dest="draw_sec", type=float, help="Tempo (segundos) do quadro para desenhar as vagas no modo draw")
    p.add_argument("--conf", type=float, help="Limiar de confiança YOLO")
    p.add_argument("--out", dest="out", help="Arquivo de saída para render")
    p.add_argument("--show-boxes", dest="show_boxes", action="store_true", help="Desenha as bounding boxes das detecções no vídeo")
    return p


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_args_with_config(args, cfg: dict) -> dict:
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})
    ui_cfg = cfg.get("ui", {})
    logic_cfg = cfg.get("logic", {})
    render_cfg = cfg.get("render", {})
    draw_cfg = cfg.get("draw", {})

    resolved = {
        "video": args.video or paths_cfg.get("video", "data/videos/parking.mp4"),
        "polygons": args.polygons or paths_cfg.get("polygons", "data/polygons.json"),
        "weights": args.weights or model_cfg.get("weights", "data/models/yolov8x.pt"),
        "conf": args.conf if args.conf is not None else float(model_cfg.get("conf_threshold", 0.2)),
        "window_name": ui_cfg.get("window_name", "Estacionamento"),
    "show_boxes": bool(getattr(args, "show_boxes", False) or ui_cfg.get("show_boxes", False)),
        "occupy_confirm_s": float(logic_cfg.get("occupy_confirm_s", 3.0)),
        "free_confirm_s": float(logic_cfg.get("free_confirm_s", 5.5)),
        "out": args.out or render_cfg.get("out", "parking_overlay.mp4"),
        "draw_seek_s": float(args.draw_sec) if getattr(args, "draw_sec", None) is not None else float(draw_cfg.get("seek_s", 57.0)),
    }
    return resolved


def cmd_draw(cfg: dict):
    video = cv2.VideoCapture(cfg["video"])
    if not video.isOpened():
        print("Não foi possível abrir o vídeo.")
        return 1

    # Posiciona no tempo solicitado (em milissegundos) para escolher um quadro nítido para desenhar
    seek_ms = max(0, int(cfg.get("draw_seek_s", 57.0) * 1000))
    video.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ret, frame = video.read()
    if not ret:
        seconds = (seek_ms / 1000.0)
        print(f"Não foi possível capturar o frame em {seconds:.2f}s. Tente outro valor com --draw-sec ou ajuste em configs/config.yaml (draw.seek_s).")
        return 1

    polys, thresholds = load_polygons(cfg["polygons"])  # pode estar vazio
    draw_polygons_interactive(frame, cfg["window_name"], polys, thresholds, cfg["polygons"])
    return 0


def process_stream(cfg: dict, writer=None):
    detector = VehicleDetector(cfg["weights"], cfg["conf"])
    polys, thresholds = load_polygons(cfg["polygons"])
    if len(polys) == 0:
        print("Nenhum polígono disponível. Use o comando 'draw' antes.")
        return 1

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.")
        return 1

    # Garante início do vídeo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # FPS para reproduzir em velocidade adequada
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    delay = max(1, int(1000 / fps))

    # Para barra de progresso no modo render (progresso)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    current_frame_idx = 0

    # Índice de frame para cálculo de tempo de vídeo (fallback)
    frame_idx_for_time = 0

    num_slots = len(polys)
    sm = OccupancyStateMachine(num_slots, cfg["occupy_confirm_s"], cfg["free_confirm_s"])

    # Configuração de exibição (apenas para watch). Mantém processamento e render no tamanho original
    created_window = False
    max_w, max_h = 1280, 720

    first = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tempo de vídeo: preferir timestamp do vídeo; se indisponível, usar frame_idx/fps
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec is not None and pos_msec > 0:
            now = float(pos_msec) / 1000.0
        else:
            now = float(frame_idx_for_time) / float(fps)
        detections = detector.detect(frame)

        detected_slots = set()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            points = bbox_checkpoints(x1, y1, x2, y2)
            idx = first_polygon_satisfying(points, polys, thresholds)
            if idx != -1:
                detected_slots.add(idx)

        for i in range(num_slots):
            sm.update_slot(i, i in detected_slots, now)

        if first:
            sm.mark_detected_occupied_on_first_frame(detected_slots, now)
            first = False

        frame_display = frame.copy()
        if cfg.get("show_boxes", False):
            draw_detection_boxes(frame_display, detections, detector.class_names)
        draw_polylines_by_state(frame_display, polys, sm.state)
        occupied = sm.count_occupied()
        draw_info_panel(frame_display, num_slots, occupied)

        if writer is not None:
            # salva no tamanho original
            writer.write(frame_display)
            # Atualiza barra de progresso no terminal
            current_frame_idx += 1
            if total_frames > 0:
                pct = min(1.0, current_frame_idx / total_frames)
                bar_len = 30
                filled = int(bar_len * pct)
                bar = "#" * filled + "-" * (bar_len - filled)
                sys.stdout.write(f"\rRenderizando: [{bar}] {pct*100:5.1f}%  ({current_frame_idx}/{total_frames})")
                sys.stdout.flush()
            else:
                # total desconhecido
                sys.stdout.write(f"\rRenderizando frames: {current_frame_idx}")
                sys.stdout.flush()
        else:
            # Redimensiona apenas para exibição (evita corte)
            h, w = frame_display.shape[:2]
            scale_x = min(1.0, max_w / float(w))
            scale_y = min(1.0, max_h / float(h))
            scale = min(scale_x, scale_y)
            if scale < 1.0:
                disp_w, disp_h = int(w * scale), int(h * scale)
                to_show = cv2.resize(frame_display, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                to_show = frame_display

            if not created_window:
                cv2.namedWindow(cfg["window_name"], cv2.WINDOW_NORMAL)
                if scale < 1.0:
                    cv2.resizeWindow(cfg["window_name"], to_show.shape[1], to_show.shape[0])
                created_window = True

            cv2.imshow(cfg["window_name"], to_show)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:
                break

        # Avança o índice de frame usado para cálculo de tempo (sempre por frame processado)
        frame_idx_for_time += 1

    cap.release()
    if writer is None:
        cv2.destroyAllWindows()
    else:
        # Finaliza linha da barra de progresso
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


def cmd_watch(cfg: dict):
    return process_stream(cfg, writer=None)


def cmd_render(cfg: dict):
    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        print("Não foi possível abrir o vídeo.")
        return 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg["out"], fourcc, fps, (width, height))
    cap.release()
    try:
        return process_stream(cfg, writer=out)
    finally:
        out.release()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = resolve_args_with_config(args, cfg)

    if args.command == "draw":
        return cmd_draw(cfg)
    elif args.command == "watch":
        return cmd_watch(cfg)
    elif args.command == "render":
        return cmd_render(cfg)
    else:
        parser.print_help()
        return 2
