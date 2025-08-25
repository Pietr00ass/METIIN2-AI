from __future__ import annotations
import time
import numpy as np
import pyautogui
from .ocr import Ocr

class Teleporter:
    def __init__(self, window_capture, use_ocr: bool = True, templates=None):
        self.win = window_capture
        self.use_ocr = use_ocr
        self.ocr = Ocr() if use_ocr else None
        self.templates = templates  # opcjonalny TemplateMatcher

    def _frame(self):
        fr = self.win.grab()
        return np.array(fr)[:, :, :3]

    def open_panel(self):
        pyautogui.hotkey('ctrl', 'x')
        time.sleep(0.2)

    def _click_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        left, top, w, h = self.win.region
        cx = int(left + (x1 + x2) / 2)
        cy = int(top + (y1 + y2) / 2)
        pyautogui.moveTo(cx, cy, duration=0)
        pyautogui.click()

    def _find_by_text(self, frame, text):
        if not self.ocr:
            return None, 0.0
        return self.ocr.find_label(frame, text)

    def _find_by_template(self, frame, key):
        if not self.templates:
            return None, 0.0
        return self.templates.find(frame, key)

    def _click_label(self, text_or_key, retries=6):
        for _ in range(retries):
            frame = self._frame()
            bbox, conf = (self._find_by_text(frame, text_or_key) if self.use_ocr else self._find_by_template(frame, text_or_key))
            if bbox is not None:
                self._click_bbox(bbox)
                return True
            time.sleep(0.1)
        return False

    def teleport(self, point_name: str, side_name: str, confirm_label: str | None = 'Teleport') -> bool:
        self.open_panel()
        ok1 = self._click_label(point_name)
        time.sleep(0.1)
        ok2 = self._click_label(side_name)
        if confirm_label:
            self._click_label(confirm_label)
        return ok1 and ok2
