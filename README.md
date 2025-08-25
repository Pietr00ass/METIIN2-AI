# Metin2 Vision Agent (Windows, Python, VS Code)

Lokalny agent wizyjny sterujący WASD i myszą: wykrywa **metiny/bossów/potwory** (YOLO), unika kolizji, potrafi **teleportować** (Ctrl+X + klik na etykietach), ma **GUI** z podglądem + overlay YOLO oraz priorytety celów ustawiane *drag & drop*.

**Brak HTTP** — wszystko lokalnie: przechwytywanie okna gry, sterowanie wejściami i CV.

## Szybki start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python gui/app.py
```

1. Wpisz fragment **tytułu okna** gry (np. `Metin2`) → `Start podglądu`.
2. (Opcjonalnie) włącz **Overlay YOLO** i wskaż `runs/detect/train/weights/best.pt`.
3. Ustaw **priorytety celów** (przeciągnij `boss/metin/potwory`).
4. Start **agenta YOLO+WASD** lub tryb **Teleportuj i poluj** (podaj `Punkt`, `Strona`, `Czas`).
5. **Panic STOP:** F12 — zwalnia wszystkie klawisze, zatrzymuje pętle.

## Pipeline danych (YOLO)

1. Zbierz nagrania: `Nagrywaj dane (5 min)` w GUI → zapisy do `data/recordings/`.
2. Ekstrakcja klatek:
   ```powershell
   python tools/extract_frames.py
   ```
   Klatki w `datasets/mt2/images/train/`.
3. Oznacz w `labelImg` (format YOLO) klasy: `metin`, `boss`, `potwory`.
4. Przygotuj `datasets/mt2/data.yaml` (jest w repo).
5. Trenuj:
   ```powershell
   python training/train_yolo.py --data datasets/mt2/data.yaml --model yolov8n.pt --epochs 50 --device cpu
   ```
   Model: `runs/detect/train/weights/best.pt`.

## Foldery

- `agent/` — logika detekcji, wybór celu wg priorytetów, WASD, unikanie, teleport.
- `recorder/` — przechwytywanie okna, nagrywanie ekranu + zdarzeń.
- `gui/` — aplikacja PySide6 z overlay YOLO, priorytetami i scenariuszem teleport+polowanie.
- `tools/` — ekstrakcja klatek z nagrań.
- `training/` — profile treningu YOLO.
- `datasets/` — miejsce na obrazy/etykiety YOLO i `data.yaml`.

> Używaj na **własnym serwerze/test-kliencie** i w zgodzie z ToS. Projekt nie obchodzi zabezpieczeń cudzych serwerów.
