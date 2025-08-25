from __future__ import annotations
import threading
import pyautogui

class KeyHold:
    def __init__(self):
        self.down = set()
        self.lock = threading.Lock()

    def press(self, key: str):
        with self.lock:
            if key not in self.down:
                pyautogui.keyDown(key)
                self.down.add(key)

    def release(self, key: str):
        with self.lock:
            if key in self.down:
                pyautogui.keyUp(key)
                self.down.remove(key)

    def release_all(self):
        with self.lock:
            for k in list(self.down):
                pyautogui.keyUp(k)
            self.down.clear()
