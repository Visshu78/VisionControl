from typing import Tuple, Callable, Optional
import numpy as np
import screen_brightness_control as sbc
import threading
import time

class BrightnessControl:
    def __init__(
        self,
        length_range: Tuple[float, float] = (50.0, 300.0),
        brightness_range: Tuple[int, int] = (0, 100),
        smooth_factor: float = 0.2,
        notify_callback: Optional[Callable[[int], None]] = None
    ):
        self.len_min, self.len_max = length_range
        self.bmin, self.bmax = brightness_range
        self.smooth_factor = float(np.clip(smooth_factor, 0.0, 0.95))
        self.notify_callback = notify_callback

        # internal state
        try:
            cur = sbc.get_brightness()
            # sbc.get_brightness can return list on multi-monitor; take first element if so
            if isinstance(cur, (list, tuple)):
                cur = cur[0]
            self.current_brightness = int(np.clip(int(cur), self.bmin, self.bmax))
        except Exception:
            # fallback
            self.current_brightness = int((self.bmin + self.bmax) // 2)

        # lock for thread-safety if user calls from multiple threads
        self._lock = threading.Lock()

    def _map_length_to_brightness(self, length: float) -> int:
        # map length in [len_min,len_max] -> brightness in [bmin,bmax]
        brightness = np.interp(length, [self.len_min, self.len_max], [self.bmin, self.bmax])
        brightness = int(np.round(np.clip(brightness, self.bmin, self.bmax)))
        return brightness

    def _apply_brightness(self, target: int):
        # smooth transition
        with self._lock:
            if self.smooth_factor <= 0:
                new_b = target
            else:
                # exponential-ish smoothing: move a fraction toward target
                new_b = int(round(self.current_brightness + (target - self.current_brightness) * self.smooth_factor))

            new_b = int(np.clip(new_b, self.bmin, self.bmax))
            if new_b == self.current_brightness:
                return  # nothing to do

            # set system brightness (may raise on unsupported platforms)
            try:
                sbc.set_brightness(new_b)
                self.current_brightness = new_b
                if self.notify_callback:
                    try:
                        self.notify_callback(new_b)
                    except Exception:
                        pass
            except Exception as e:
                # fail silently but keep internal state consistent
                # you can log or raise depending on your needs
                # print("Brightness set failed:", e)
                self.current_brightness = new_b

    def update_from_length(self, length: float):
        
        target = self._map_length_to_brightness(length)
        self._apply_brightness(target)

    def update_from_landmarks(self, thumb: Tuple[float, float], index: Tuple[float, float]):
        
        dx = thumb[0] - index[0]
        dy = thumb[1] - index[1]
        length = float(np.hypot(dx, dy))
        self.update_from_length(length)

    def get_current_brightness(self) -> int:
        with self._lock:
            return int(self.current_brightness)

    def set_direct(self, brightness: int):
        b = int(np.clip(int(brightness), self.bmin, self.bmax))
        self._apply_brightness(b)

# Example usage:
if __name__ == "__main__":
    # Demonstration (no camera). Replace the simulated lengths with real values from your hand tracker.
    def cb(b):
        print("Brightness now:", b, "%")

    ctrl = BrightnessControl(length_range=(50, 300), brightness_range=(0, 100), smooth_factor=0.25, notify_callback=cb)

    # Simulate a few lengths coming from your hand-tracker:
    simulated_lengths = [60, 80, 120, 200, 280, 320, 200, 100, 60]
    for L in simulated_lengths:
        ctrl.update_from_length(L)
        time.sleep(0.25)  # simulate frame rate / update rate