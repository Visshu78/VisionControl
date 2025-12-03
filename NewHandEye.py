# mouse_hand_control.py
# Minimal mouse + eye-click controller (no camera, no OpenCV).
# Provide normalized or pixel coordinates from your tracker and/or eye-closure booleans/metrics.

import time
import threading
import pyautogui
from typing import Optional, Tuple

class MouseHandEyeController:
    def __init__(
        self,
        screen_size: Optional[Tuple[int,int]] = None,
        frame_size: Optional[Tuple[int,int]] = None,
        smooth: float = 0.0,
        click_threshold: float = 0.05,
        double_click_threshold: float = 0.1,
        eye_metric_threshold: float = 0.010
    ):
        """
        screen_size: (width,height) of screen. If None, fetched from pyautogui.
        frame_size: (frame_w, frame_h) - if you will pass pixel coords; if you pass normalized coords use None.
        smooth: 0..1 fraction for smoothing cursor (0 = immediate).
        click_threshold: seconds eye must remain closed to register a click.
        double_click_threshold: seconds between two clicks to treat as double click.
        eye_metric_threshold: if using eye metric (upper_y - lower_y), metric < threshold means closed.
        """
        if screen_size:
            self.screen_w, self.screen_h = screen_size
        else:
            s = pyautogui.size()
            self.screen_w, self.screen_h = s.width, s.height

        self.frame_size = frame_size  # if provided, used to map pixel coords to screen
        self.smooth = max(0.0, min(0.95, smooth))

        # click timing
        self.click_threshold = click_threshold
        self.double_click_threshold = double_click_threshold
        self.eye_metric_threshold = eye_metric_threshold

        # internal state
        self._lock = threading.Lock()
        self._ploc = (0,0)   # previous location (screen coords)
        self._cloc = (0,0)   # current location (screen coords)

        self._eye_closed_start = 0.0
        self._is_clicking = False
        self._click_times = []  # timestamps of clicks for double-click detection

        # Controller ON by default
        self.enabled = True

    # ---------- Cursor movement ----------
    def _map_to_screen(self, x: float, y: float, normalized: bool) -> Tuple[int,int]:
        """
        If normalized=True -> x,y expected in [0..1], origin top-left.
        If normalized=False -> x,y are pixel coords in frame_size, frame_size must be set.
        Returns (sx, sy) screen coordinates (ints).
        """
        if normalized:
            sx = int(x * self.screen_w)
            sy = int(y * self.screen_h)
            return sx, sy
        else:
            if not self.frame_size:
                raise ValueError("frame_size must be set if passing pixel coords (normalized=False).")
            fw, fh = self.frame_size
            sx = int(x * (self.screen_w / fw))
            sy = int(y * (self.screen_h / fh))
            return sx, sy

    def move_cursor(self, x: float, y: float, normalized: bool = True):
        """
        Move cursor to the coordinate received from hand-tracker.
        x,y: either normalized (0..1) if normalized=True, or pixel coords if normalized=False (requires frame_size).
        """
        if not self.enabled:
            return

        sx, sy = self._map_to_screen(x, y, normalized)

        with self._lock:
            if self.smooth <= 0:
                self._cloc = (sx, sy)
            else:
                px, py = self._ploc
                nx = int(round(px + (sx - px) * self.smooth))
                ny = int(round(py + (sy - py) * self.smooth))
                self._cloc = (nx, ny)

            self._ploc = self._cloc

        # Move OS cursor (pyautogui)
        try:
            pyautogui.moveTo(self._cloc[0], self._cloc[1], duration=0)
        except Exception:
            # silent fail if environment doesn't allow mouse control
            pass

    # ---------- Eye / Click logic ----------
    def _register_click(self, click_type: str = "left"):
        """
        click_type: 'left' or 'right'
        Handles single vs double click using timestamps.
        """
        t = time.time()
        self._click_times.append(t)

        # keep only last 2
        if len(self._click_times) > 2:
            self._click_times = self._click_times[-2:]

        # check double click
        if len(self._click_times) >= 2 and (self._click_times[-1] - self._click_times[-2]) <= self.double_click_threshold:
            try:
                pyautogui.doubleClick()
            except Exception:
                pass
            self._click_times.clear()
            return

        # Otherwise single click - respect click_type
        try:
            if click_type == "left":
                pyautogui.click()
            elif click_type == "right":
                pyautogui.click(button='right')
            else:
                pyautogui.click()
        except Exception:
            pass

    def update_eyes(self, left_closed: bool, right_closed: bool):
        """
        Call this every frame with booleans indicating whether each eye is considered closed.
        - one-eye closed (exactly one True) => RIGHT click after holding for click_threshold.
        - both eyes closed (both True) => LEFT click after holding for click_threshold.
        """
        now = time.time()

        # determine if any eye is closed
        any_closed = left_closed or right_closed
        both_closed = left_closed and right_closed
        one_closed = (left_closed ^ right_closed)

        if any_closed:
            # start or continue counting closed duration
            if self._eye_closed_start == 0.0:
                self._eye_closed_start = now
            elapsed = now - self._eye_closed_start

            if elapsed >= self.click_threshold and not self._is_clicking:
                # decide click type
                if both_closed:
                    self._register_click("left")
                elif one_closed:
                    # original spec: one-eye blink -> right click
                    self._register_click("right")
                # mark clicked and reset timer so user must open eyes to re-trigger
                self._is_clicking = True
                self._eye_closed_start = 0.0
        else:
            # eyes open -> reset
            self._eye_closed_start = 0.0
            self._is_clicking = False

    def update_eye_metrics(self, left_metric: float, right_metric: float):
        """
        If your tracker gives a numeric metric per eye (e.g., upper_y - lower_y),
        call this and it will compare with eye_metric_threshold to derive closed/open.
        Smaller metric -> more closed.
        """
        left_closed = left_metric < self.eye_metric_threshold
        right_closed = right_metric < self.eye_metric_threshold
        self.update_eyes(left_closed, right_closed)

    # ---------- Utility ----------
    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_frame_size(self, frame_w: int, frame_h: int):
        """Set frame size if you want to pass pixel coords (normalized=False)."""
        self.frame_size = (frame_w, frame_h)

    def set_screen_size(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

    def get_cursor(self) -> Tuple[int,int]:
        return self._cloc

# Example usage:
if __name__ == "__main__":
    # Simulated integration example (no camera). Replace this with calls from your hand-tracker:
    ctrl = MouseHandEyeController(smooth=0.15, click_threshold=0.25, double_click_threshold=0.15)

    # Example: moving cursor with normalized coords (0..1)
    simulated_positions = [(0.1,0.1),(0.2,0.15),(0.3,0.2),(0.5,0.5)]
    for pos in simulated_positions:
        ctrl.move_cursor(pos[0], pos[1], normalized=True)
        time.sleep(0.05)

    # Example: simulating eye metrics (lower value = closed)
    # Keep left eye closed for longer than threshold -> triggers right click
    ctrl.update_eye_metrics(left_metric=0.005, right_metric=0.03)  # closed left
    time.sleep(0.3)
    ctrl.update_eye_metrics(left_metric=0.005, right_metric=0.03)
    time.sleep(0.1)
    ctrl.update_eye_metrics(left_metric=0.05, right_metric=0.05)  # open -> reset

    # Both eyes closed -> left click
    ctrl.update_eye_metrics(left_metric=0.005, right_metric=0.006)
    time.sleep(0.3)
    ctrl.update_eye_metrics(left_metric=0.005, right_metric=0.006)
    time.sleep(0.1)
    ctrl.update_eye_metrics(left_metric=0.06, right_metric=0.06)
