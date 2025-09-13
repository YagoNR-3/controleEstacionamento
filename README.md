## Controle de Estacionamento (YOLOv8)

Este projeto detecta vagas ocupadas e livres em um estacionamento usando visão computacional.
Você desenha as vagas uma única vez sobre uma imagem do vídeo e o sistema mostra, em tempo real,
quantas estão ocupadas e quantas estão livres, além de gerar um vídeo com essas informações.

### O que você precisa
- Python 3.10 ou superior
- Um vídeo em (por exemplo: `data/parking.mp4`)
- Um arquivo de pesos do YOLOv8 (por exemplo: `data/models/yolov8x.pt`)
- Você pode baixar pesos do YOLOv8 no site da Ultralytics: [Página de modelos YOLOv8](https://docs.ultralytics.com/models/yolov8/)

### Como começar (passo a passo)
1) Instale as dependências:
```bash
pip install -r requirements.txt
```

2) Coloque seus arquivos:
- Vídeo em: `data/` (ex.: `data/parking.mp4`)
- Pesos YOLO em: `data/models/` (ex.: `data/models/yolov8x.pt`)

3) Desenhe as vagas (apenas na primeira vez):
```bash
python scripts/parking_monitor.py draw --video data/parking.mp4
```
- Clique: adiciona ponto
- f: finaliza o polígono (não salva)
- Espaço: finaliza e salva o polígono
- c: remove o último polígono salvo
- Backspace: desfaz o último ponto
- 1: marca o último polígono com sensibilidade maior se necessário (threshold=1)
- q/ESC: sair

Observação sobre o quadro escolhido no draw
- Neste projeto, escolhemos capturar o quadro do segundo 57 do vídeo para desenhar as vagas, pois nesse momento as vagas estavam mais visíveis e com menos oclusões. Esse tempo é apenas um exemplo e varia de acordo com seu vídeo: escolha um instante em que as vagas estejam bem enquadradas e claramente visíveis.
- Como alterar o instante do quadro:
  - Pela linha de comando, use a opção --draw-sec (em segundos):
    ```bash
    python scripts/parking_monitor.py draw --video data/parking.mp4 --draw-sec 42
    ```
    O exemplo acima posiciona o vídeo em 42 segundos antes de capturar o frame para desenhar.

  - Pelo arquivo de configuração, ajuste a chave draw.seek_s em configs/config.yaml:
    ```yaml
    draw:
      seek_s: 57.0
    ```
    Defina o valor (em segundos) que melhor representa um quadro limpo para o desenho dos polígonos. A opção de CLI (--draw-sec) sobrepõe o valor do config.

4) Monitoramento (modo "watch"):
```bash
python scripts/parking_monitor.py watch --video data/parking.mp4
```
- O monitoramento contínuo (por frame) é exibido no terminal, com os logs/inferências do YOLO em tempo real.
- Na janela de vídeo, é mostrado apenas um frame com as detecções atuais (não há reprodução contínua do vídeo neste modo).
- Para obter um vídeo com as informações desenhadas em todos os frames, use o passo 5 (render).

5) Gere um vídeo com as informações de detecção e controle:
```bash
# usa o nome padrão do config (parking_overlay.mp4)
python scripts/parking_monitor.py render --video data/parking.mp4
```
- Durante o render, uma barra de progresso aparece no terminal indicando a porcentagem e os frames processados.
- O nome padrão do arquivo de saída pode ser alterado em `configs/config.yaml` (chave `render.out`).
- As caixas de detecção (bounding boxes) podem ser desenhadas. Ative pela CLI com `--show-boxes` ou ajuste `ui.show_boxes` (true/false) em `configs/config.yaml`.

Se preferir, você pode ajustar caminhos e parâmetros no arquivo de configuração `configs/config.yaml`.

### Sobre os vídeos do repositório
- O vídeo demonstrado no repositório foi gerado pelo projeto antigo que está em `docs/legacy/`. O projeto atual não terá exatamente o mesmo visual no output (por exemplo, cores das bounding boxes, ritmo de aparecer/desaparecer elementos). Essas diferenças foram escolhas deliberadas para deixar a demonstração mais dinâmica e visualmente agradável no exemplo antigo.

#### Sobre o “projeto antigo” x “projeto atual”
- É o mesmo projeto em essência: YOLOv8 para detectar veículos + verificação por checkpoints (5 pontos por caixa, com limiar por vaga) + máquina de estados para confirmar entrada/saída.
- Por que criar uma nova estrutura em vez de seguir com os scripts antigos:
  - Modularização: saímos de scripts monolíticos em `docs/legacy/` para módulos organizados em `src/parking_monitor/` (detector, polígonos, FSM de ocupação, visualização). Isso facilita manutenção, testes e evolução.
  - CLI unificada: comandos `draw`, `watch` e `render` centralizados em `scripts/parking_monitor.py`, com flags padrão e ajuda (`--help`).
  - Config central: `configs/config.yaml` reúne caminhos e parâmetros (vídeo, pesos, confiança, tempos da FSM, UI), evitando caminhos hardcoded dos scripts antigos.
  - UX/documentação: barra de progresso no render, ajuda on-screen no draw, opção de mostrar bounding boxes, escolha do quadro do draw (`--draw-sec`).
  - Tempo de confirmação consistente: no atual, os atrasos da FSM são medidos em tempo de VÍDEO; nos scripts antigos eram por tempo de relógio, o que podia distorcer a percepção no render.

Em resumo: a base lógica é a mesma, mas a nova estrutura é mais organizada, configurável e amigável. As diferenças visuais entre a demo antiga e o output atual foram decisões de UX para tornar a demonstração mais clara e agradável, e podem ser ajustadas via config/flags.

### Desempenho e tempo de processamento
- O tempo de execução depende principalmente do modelo YOLO e do hardware.
- Em CPU, o modelo grande (`yolov8x.pt`) é lento por natureza (centenas de ms por frame). Isso faz o render demorar em vídeos longos.
- No modo `watch`, o terminal mostra as inferências em tempo real, mas a janela pode não reproduzir continuamente o vídeo dependendo da configuração atual.

Como acelerar (sem alterar código):
- Use um modelo menor: baixe `yolov8n.pt` (nano) ou `yolov8s.pt` (small) e rode.
- Use vídeos mais curtos ao testar.

### Estrutura do projeto
```
.
├─ src/parking_monitor/          # Código-fonte do sistema (módulos)
│  ├─ __init__.py
│  ├─ cli.py                     # Interface de linha de comando (CLI)
│  ├─ detector.py                # Carregamento e inferência do YOLO
│  ├─ polygons.py                # Desenho e salvamento das vagas (polígonos)
│  ├─ occupancy.py               # Lógica de ocupação com atraso/estabilidade
│  └─ visualization.py           # Elementos visuais (overlays, painel)
├─ scripts/
│  └─ parking_monitor.py         # Ponto de entrada do CLI
├─ data/
│  ├─ parking.mp4                # Coloque seus vídeos aqui
│  ├─ models/                    # Coloque os pesos do YOLO aqui
│  ├─ polygons.json              # Vagas salvas (criado após desenhar)
│  └─ README.md
├─ configs/
│  └─ config.yaml                # Caminhos e parâmetros padrão
├─ docs/
│  └─ legacy/                    # Scripts antigos (apenas referência)
├─ requirements.txt
└─ README.md
```

### Como funciona (em poucas palavras)
- O detector identifica veículos no vídeo.
- Para cada veículo, o sistema checa se ele “entra” em alguma vaga desenhada.
- Uma máquina de estados evita oscilações: exige um tempo mínimo para confirmar que uma vaga ficou ocupada ou ficou livre.
- A interface mostra total, ocupadas, livres e uma barra de ocupação.