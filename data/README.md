Esta pasta guarda os arquivos de dados locais.

Coloque aqui:
- : seus vídeos de entrada (ex.: `parking.mp4`)
- `models/`: os pesos do YOLO (ex.: `yolov8x.pt`)

O arquivo `polygons.json` guarda as vagas desenhadas e é criado automaticamente quando você usa o comando:
```bash
python scripts/parking_monitor.py draw --video data/parking.mp4
```
