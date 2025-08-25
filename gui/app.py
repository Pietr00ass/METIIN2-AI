from __future__ import annotations
import sys, time, threading, subprocess
from pathlib import Path

import numpy as np
import cv2
import pyautogui
from PySide6 import QtCore, QtWidgets, QtGui
from pynput import keyboard

from recorder.window_capture import WindowCapture
from agent.detector import ObjectDetector
from agent.hunt_destroy import HuntDestroy
from agent.teleport import Teleporter
from agent.wasd import KeyHold

pyautogui.FAILSAFE = False  # wyłącz krawędziowy failsafe

class PreviewWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(np.ndarray)
    status = QtCore.Signal(str)

    def __init__(self, title_substr: str):
        super().__init__()
        self.title = title_substr
        self._stop = False
        self._det = None
        self._overlay = False
        self._classes = None

    def configure_overlay(self, model_path: str | None, classes: list[str] | None, enabled: bool):
        self._overlay = enabled
        self._classes = classes
        if enabled and model_path:
            try:
                self._det = ObjectDetector(model_path, classes)
                self.status.emit("Overlay YOLO aktywny.")
            except Exception as e:
                self.status.emit(f"Błąd YOLO: {e}")
                self._det = None
        else:
            self._det = None

    def run(self):
        try:
            cap = WindowCapture(self.title)
            self.status.emit("Szukam okna…")
            cap.locate()
            self.status.emit("Znaleziono okno. Podgląd działa.")
            while not self._stop:
                fr = cap.grab()
                frame = np.array(fr)[:, :, :3]
                if self._overlay and self._det is not None:
                    try:
                        dets = self._det.infer(frame)
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d['bbox'])
                            color = (0, 0, 255)
                            if d['name'] == 'boss':
                                color = (0, 215, 255)
                            elif d['name'] == 'potwory':
                                color = (255, 128, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{d['name']} {d['conf']:.2f}", (x1, max(12, y1-6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as e:
                        self.status.emit(f"Overlay YOLO błąd: {e}")
                self.frame_ready.emit(frame)
                self.msleep(33)
        except Exception as e:
            self.status.emit(f"Błąd podglądu: {e}")

    def stop(self):
        self._stop = True

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metin2 Vision Agent – Panel")
        self.resize(1200, 780)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout(); layout.addLayout(left, 1)

        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("Fragment tytułu okna (np. Metin2)")
        left.addWidget(QtWidgets.QLabel("Tytuł okna:"))
        left.addWidget(self.title_edit)

        self.model_path = QtWidgets.QLineEdit("runs/detect/train/weights/best.pt")
        self.classes_edit = QtWidgets.QLineEdit("metin,boss,potwory")
        left.addWidget(QtWidgets.QLabel("Ścieżka modelu YOLO:"))
        left.addWidget(self.model_path)
        left.addWidget(QtWidgets.QLabel("Klasy obiektów:"))
        left.addWidget(self.classes_edit)

        left.addWidget(QtWidgets.QLabel("Priorytety (przeciągnij aby zmienić):"))
        self.prio_list = QtWidgets.QListWidget()
        self.prio_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        for name in ["boss", "metin", "potwory"]:
            self.prio_list.addItem(QtWidgets.QListWidgetItem(name))
        left.addWidget(self.prio_list)

        self.deadzone = QtWidgets.QDoubleSpinBox(); self.deadzone.setRange(0.0, 0.5); self.deadzone.setSingleStep(0.01); self.deadzone.setValue(0.05)
        self.desired_w = QtWidgets.QDoubleSpinBox(); self.desired_w.setRange(0.02, 0.5); self.desired_w.setSingleStep(0.01); self.desired_w.setValue(0.12)
        form = QtWidgets.QFormLayout()
        form.addRow("Deadzone X:", self.deadzone)
        form.addRow("Desired box W:", self.desired_w)
        left.addLayout(form)

        self.overlay_chk = QtWidgets.QCheckBox("Overlay YOLO na podglądzie")
        self.overlay_chk.setChecked(True)
        left.addWidget(self.overlay_chk)

        left.addWidget(QtWidgets.QLabel("Teleportacja:"))
        self.tp_point = QtWidgets.QLineEdit(); self.tp_point.setPlaceholderText("Nazwa punktu (OCR lub template)")
        self.tp_side  = QtWidgets.QLineEdit(); self.tp_side.setPlaceholderText("Strona/mapa")
        self.tp_minutes = QtWidgets.QSpinBox(); self.tp_minutes.setRange(1, 180); self.tp_minutes.setValue(10)
        form2 = QtWidgets.QFormLayout()
        form2.addRow("Punkt:", self.tp_point)
        form2.addRow("Strona:", self.tp_side)
        form2.addRow("Czas (min):", self.tp_minutes)
        left.addLayout(form2)

        self.btn_preview = QtWidgets.QPushButton("Start podglądu")
        self.btn_record  = QtWidgets.QPushButton("Nagrywaj dane (5 min)")
        self.btn_agent   = QtWidgets.QPushButton("Start agenta (YOLO + WASD)")
        self.btn_tp_hunt = QtWidgets.QPushButton("Teleportuj i poluj")
        self.btn_stop    = QtWidgets.QPushButton("STOP (F12)")
        self.btn_train   = QtWidgets.QPushButton("Trenuj YOLO (CLI)")
        for b in [self.btn_preview, self.btn_record, self.btn_agent, self.btn_tp_hunt, self.btn_stop, self.btn_train]:
            left.addWidget(b)

        left.addStretch(1)
        self.status_label = QtWidgets.QLabel("Gotowy.")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)

        right = QtWidgets.QVBoxLayout(); layout.addLayout(right, 2)
        self.video = QtWidgets.QLabel(); self.video.setMinimumSize(860, 480)
        self.video.setStyleSheet("background:#222; border:1px solid #444")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self.video)

        self.preview_thread: PreviewWorker | None = None
        self.agent_thread: threading.Thread | None = None
        self._panic = False
        self._hotkey_listener = None

        self.btn_preview.clicked.connect(self.toggle_preview)
        self.btn_record.clicked.connect(self.record_data)
        self.btn_agent.clicked.connect(self.start_agent)
        self.btn_tp_hunt.clicked.connect(self.start_tp_and_hunt)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_train.clicked.connect(self.train_yolo_cli)

        self.start_hotkey_listener()

    def current_priority(self):
        return [self.prio_list.item(i).text() for i in range(self.prio_list.count())]

    def set_status(self, txt: str):
        self.status_label.setText(txt)

    def show_frame(self, frame):
        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video.width(), self.video.height(), QtCore.Qt.KeepAspectRatio)
        self.video.setPixmap(pix)

    def toggle_preview(self):
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.stop(); self.preview_thread.wait(); self.preview_thread = None
            self.btn_preview.setText("Start podglądu"); self.set_status("Podgląd zatrzymany.")
            return
        title = self.title_edit.text().strip()
        if not title:
            self.set_status("Podaj fragment tytułu okna.")
            return
        self.preview_thread = PreviewWorker(title)
        self.preview_thread.frame_ready.connect(self.show_frame)
        self.preview_thread.status.connect(self.set_status)
        classes = [c.strip() for c in self.classes_edit.text().split(',') if c.strip()]
        self.preview_thread.configure_overlay(self.model_path.text().strip(), classes, self.overlay_chk.isChecked())
        self.preview_thread.start()
        self.btn_preview.setText("Stop podglądu")

    def record_data(self):
        from recorder.capture import record_session
        title = self.title_edit.text().strip()
        if not title:
            self.set_status("Podaj fragment tytułu okna.")
            return
        wc = WindowCapture(title); wc.locate(); wc.update_region()
        l, t, w, h = wc.region
        def job():
            try:
                self.set_status("Nagrywanie 5 min…")
                record_session('data/recordings', region=(l, t, w, h), fps=15, duration_sec=300)
                self.set_status("Nagrywanie zakończone. Użyj narzędzia 'extract_frames'.")
            except Exception as e:
                self.set_status(f"Błąd nagrywania: {e}")
        threading.Thread(target=job, daemon=True).start()

    def build_cfg(self):
        title = self.title_edit.text().strip()
        classes = [c.strip() for c in self.classes_edit.text().split(',') if c.strip()]
        prio = self.current_priority()
        return {
            'window': {'title_substr': title},
            'controls': {'keys': {'forward': 'w', 'left': 'a', 'back': 's', 'right': 'd'}},
            'detector': {
                'model_path': self.model_path.text().strip(),
                'classes': classes,
                'conf_thr': 0.5, 'iou_thr': 0.45
            },
            'policy': {
                'deadzone_x': float(self.deadzone.value()),
                'desired_box_w': float(self.desired_w.value()),
                'key_repeat_ms': 60
            },
            'stuck': {'flow_window': 0.8, 'min_flow_mag': 0.7, 'rotate_ms_on_stuck': 250},
            'priority': prio
        }

    def start_agent(self):
        cfg = self.build_cfg()
        def run():
            try:
                agent = HuntDestroy(cfg, WindowCapture(cfg['window']['title_substr']))
                assert agent.win.locate()
                while not self._panic:
                    agent.step(); time.sleep(1/15)
            except Exception as e:
                self.set_status(f"Błąd agenta: {e}")
        self._panic = False
        self.agent_thread = threading.Thread(target=run, daemon=True)
        self.agent_thread.start(); self.set_status("Agent YOLO+WASD uruchomiony.")

    def start_tp_and_hunt(self):
        point = self.tp_point.text().strip(); side = self.tp_side.text().strip(); minutes = int(self.tp_minutes.value())
        if not point or not side:
            self.set_status("Uzupełnij punkt i stronę teleportacji.")
            return
        cfg = self.build_cfg()
        def run():
            try:
                win = WindowCapture(cfg['window']['title_substr']); assert win.locate()
                tp = Teleporter(win, use_ocr=True)
                ok = tp.teleport(point, side)
                if not ok:
                    self.set_status("Nie udało się kliknąć elementów w panelu teleportu (sprawdź OCR/templates)")
                hd = HuntDestroy(cfg, win)
                t_end = time.time() + minutes * 60
                while time.time() < t_end and not self._panic:
                    hd.step(); time.sleep(1/15)
                self.set_status("Zakończono 'Teleportuj i poluj'.")
            except Exception as e:
                self.set_status(f"Błąd teleport+poluj: {e}")
        self._panic = False
        self.agent_thread = threading.Thread(target=run, daemon=True)
        self.agent_thread.start(); self.set_status("Teleportuję i poluję…")

    def stop_all(self):
        self._panic = True
        try:
            KeyHold().release_all()
        except Exception:
            pass
        if self.preview_thread and self.preview_thread.isRunning():
            self.preview_thread.stop(); self.preview_thread.wait(); self.preview_thread = None
            self.btn_preview.setText("Start podglądu")
        self.set_status("STOP – wszystkie klawisze zwolnione.")

    def train_yolo_cli(self):
        cmd = [sys.executable, '-m', 'ultralytics', 'detect', 'train',
               'data=datasets/mt2/data.yaml', 'model=yolov8n.pt', 'imgsz=640', 'epochs=50', 'batch=16']
        def job():
            try:
                self.set_status("Trening YOLO – start…")
                subprocess.run(cmd, check=True)
                self.set_status("Trening zakończony. Wybierz runs/detect/train/weights/best.pt")
            except Exception as e:
                self.set_status(f"Błąd treningu: {e}")
        threading.Thread(target=job, daemon=True).start()

    def start_hotkey_listener(self):
        def on_press(key):
            try:
                if key == keyboard.Key.f12:
                    self.stop_all()
            except Exception:
                pass
        self._hotkey_listener = keyboard.Listener(on_press=on_press)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
