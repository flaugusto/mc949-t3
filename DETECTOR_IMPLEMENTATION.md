## Navegação Assistida – Implementação do detector.py

### Resumo
O arquivo `detector.py` implementa um pipeline em tempo quase real para: detecção de objetos relevantes à navegação, estimativa de proximidade a partir de profundidade monocular, seleção do alvo mais importante à frente, lógica simples de risco/colisão com mensagens de ação por TTS em PT‑BR, e visualização/salvamento de vídeo com UI. Também grava o vídeo final com áudio das narrações automaticamente (via ffmpeg).

### Requisitos do T3 atendidos
- R1 – Detecção de obstáculos: detector YOLOv8n identifica classes relevantes e um fallback de profundidade identifica pilastras.
- R2 – Caminhada sem colisões: há uma regra de risco que aciona alertas “Pare”/“Desvie à esquerda/direita” conforme proximidade e zona de perigo.
- R3 – Narração da classe: TTS PT‑BR narra a classe em português e a faixa de distância do alvo mais relevante; há supressão de spam e prioridade para alertas.
- R4 – Estimativa de distância: baseada em mapa de profundidade monocular (MiDaS), com faixas qualitativas (muito perto, perto, médio, longe).

Requisitos técnicos mínimos:
- Detecção/segmentação: YOLOv8n (Ultralytics).
- Distância: MiDaS_small (depth monocular) com pós‑processamento leve.
- Lógica de no‑collision: heurística com ROI/zona de perigo e thresholds de proximidade.
- TTS PT‑BR: usando `espeak-ng`, com rate‑limiting e cooldown de mensagens repetidas.
- Medições: overlay com FPS e latência de fala (captura→TTS), além de gravação do vídeo com UI.

### Arquitetura do pipeline
1) Captura de vídeo (`cv2.VideoCapture`).
2) Profundidade (MiDaS): gera `depth` e `depth_vis`; aplica blur leve e normalização por percentis 1–99 para robustez visual.
3) Detecção (YOLOv8n): retorna bboxes + classe + confiança.
4) Filtro de classes e limiares: mantém apenas classes úteis e aplica thresholds específicos (confiança, área, razão de aspecto quando aplicável).
5) Fallback por profundidade: detecta pilastras como componentes “estreitos e altos” próximos no mapa de profundidade (pseudo‑classe `pillar`).
6) Proximidade por bbox: calcula mediana do depth na bbox; normaliza e inverte (0=longe, 1=mais perto); mapeia para faixas qualitativas.
7) Seleção do alvo: um score combina proximidade (suavizada), área normalizada, ROI central e peso por classe; escolhe apenas 1 alvo por quadro.
8) Risco/colisão: na zona de perigo, “Pare” quando muito próximo; caso contrário “Desvie” para lado oposto do alvo quando próximo.
9) TTS: mensagens de alerta têm prioridade; rate‑limiting global, cooldown para repetir o mesmo alerta, execução não bloqueante.
10) UI e gravação: desenha bounding boxes, labels PT‑BR, faixa de distância e valores; exibe FPS/latência; salva vídeo (com opção lado‑a‑lado Detection|Depth) e muxa o áudio ao final.

### Modelos e classes alvo
- Detector: `ultralytics.YOLO("yolov8n.pt")` (leve, adequado a CPU/GPU modestas).
- Profundidade: `MiDaS_small` via `torch.hub`.
- Classes mantidas e mapeamento para PT‑BR:
  - bicycle → bicicleta
  - motorbike / motorcycle → moto
  - bench → banco
  - fire hydrant → hidrante
  - pillar (pseudo‑classe via profundidade) → pilastra

Prioridade por classe (peso no score): pilastra > moto > bicicleta > banco > hidrante.

### Estimativa de proximidade e faixas
- Proximidade `prox ∈ [0,1]`: 1 = mais perto; 0 = mais longe.
- Cálculo: mediana do depth na bbox → normalização pelos percentis 1–99 do quadro → inversão (1 − normalizado).
- Faixas textuais (ajustáveis):
  - muito perto: prox ≥ 0.80
  - perto: 0.50 ≤ prox < 0.80
  - médio: 0.20 ≤ prox < 0.50
  - longe: prox < 0.20
- Suavização temporal: EMA (α=0.6) por “chave” (classe + bbox discretizada) reduz jitter.

### ROI, score e seleção do alvo
- ROI central: `x ∈ [0.30,0.70]·W`, `y ∈ [0.20,0.90]·H`.
- Zona de perigo: `x ∈ [0.40,0.60]·W`, `y ∈ [0.60,0.95]·H`.
- Score do alvo por quadro:
  - `score = 0.5·prox_suav + 0.25·area_norm + 0.15·roi_bonus + 0.10·peso_classe`.
- Seleciona apenas 1 alvo (maior score) para narrar/exibir/avaliar risco.

### Lógica de risco e mensagens de ação
- Se alvo dentro da zona de perigo e `prox ≥ 0.85`: ação “Pare” (prioridade alta).
- Senão, se `prox ≥ 0.60`: ação “Desvie” para o lado oposto da posição horizontal do alvo.
- Sem ação abaixo desses limites para evitar spam.

### TTS em PT‑BR e anti‑spam
- TTS: `espeak-ng -v pt-br -s 170` executado de forma não bloqueante (subprocesso).
- Rate‑limiting global: janela mínima entre mensagens.
- Cooldown por tipo: alerta pode interromper classe, mas repetir o mesmo alerta exige intervalo (ex.: 2.5 s).
- A mensagem de classe usa rótulo PT‑BR e faixa (“muito perto”, “perto”, “médio”, “longe”).

### Visualização e gravação de vídeo + áudio final
- Overlays: bbox, classe PT‑BR, faixa/valor de proximidade, FPS e latência até a última fala.
- Salvamento:
  - Grava `output_raw.mp4` a `OUTPUT_FPS` (ex.: 12.0) com frames já com UI; opção `SAVE_SIDEBYSIDE` para juntar `Detection | Depth`.
  - Para cada fala, gera WAV em `tts_audio/` com timestamp alinhado ao número de frames gravados.
  - Ao final, `ffmpeg` muxa os WAVs no vídeo e gera `output.mp4` com áudio embutido.

### Performance e robustez
- Depth decimation: calcula profundidade a cada `DEPTH_EVERY_N` quadros (ex.: 2) e reutiliza nos intermediários.
- Blur leve no depth: `GaussianBlur(5×5)` para reduzir ruído.
- TTS não bloqueante: evita travar o loop.
- FPS de saída controlado (`OUTPUT_FPS`) para sincronizar melhor labels e áudio e evitar aceleração.

### Parâmetros ajustáveis principais (no topo do arquivo)
- Detecção: `DETECT_CONF`, `DETECT_IOU`, `DETECT_IMGSZ`.
- Mapeamento de classes: `TARGET_CLASS_MAP`, `CLASS_WEIGHTS`.
- Thresholds por classe: `MIN_CONF_PER_CLASS`, áreas mínimas, janelas de razão de aspecto.
- Profundidade/Proximidade: `BANDS_FOR_LABEL`, `EMA_ALPHA`, filtros.
- ROI e zona de perigo: `ROI_X`, `ROI_Y`, `DANGER_X`, `DANGER_Y`.
- Risco: `BAND_VERY_CLOSE`, `BAND_CLOSE`.
- TTS/anti‑spam: `GLOBAL_SPEAK_INTERVAL`, `ALERT_COOLDOWN`, `CLASS_COOLDOWN`, `SAME_ACTION_INTERVAL`.
- Vídeo/áudio: `SAVE_OUTPUT`, `SAVE_SIDEBYSIDE`, `OUTPUT_FPS`, `RAW_OUTPUT_PATH`, `FINAL_OUTPUT_PATH`, `SAVE_AUDIO`, `MUX_AUDIO_TO_VIDEO`, `AUDIO_DIR`.
- Fallback pilastra: `PROX_NEAR_THRESHOLD`, `PILLAR_MIN_AR`, `PILLAR_MIN_AREA_FRAC`.

### Como executar
1) Dependências (Ubuntu/WSL2):
   - `sudo apt update && sudo apt install -y ffmpeg espeak-ng`
   - `pip install ultralytics torch opencv-python numpy`
2) Coloque um vídeo em `video.mp4` ou ajuste o caminho no final do arquivo.
3) Rode: `python detector.py`.
4) Saída:
   - Visualização em janelas `Detection` e `Depth`.
   - Vídeo final com áudio em `output.mp4` e bruto em `output_raw.mp4`.

### Limitações conhecidas
- Proximidade é relativa (profundidade monocular normalizada). Para métricas absolutas (metros), seria necessário calibração/escala adicional ou SLAM/estéreo.
- Fallback de pilastra usa heurísticas no depth; pode falhar em texturas/iluminação desafiadoras.
- Latência e FPS dependem do hardware (CPU/GPU) e da resolução do vídeo.

### Próximos passos sugeridos
- Adicionar tracking (ByteTrack/OC‑SORT) para estabilidade temporal das bboxes.
- Filtro temporal das distâncias e classes com Kalman/média móvel sobre o alvo.
- Mapa de risco lateral (esquerda/centro/direita) com recomendações mais ricas.
- Priorização dinâmica por contexto (ex.: dar mais peso a objetos móveis próximos).
- Medir e registrar métricas quantitativas (FPS, latência ponta‑a‑ponta, acertos de alerta) conforme seção de avaliação do T3.


