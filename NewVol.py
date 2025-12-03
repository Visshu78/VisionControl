# volume_control.py
# Pure volume control module (no camera / no OpenCV).
# Input: either a precomputed finger-distance (length) or two landmark coords.
# Maps length -> system volume and sets it (Windows: pycaw; fallback: try scalar/dB; Linux: uses amixer if available).

from typing import Tuple, Callable, Optional
import numpy as np
import threading
import time

# Try to import pycaw (Windows). If unavailable, we fall back to a no-op or CLI approach.
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _PYCAW_AVAILABLE = True
except Exception:
    _PYCAW_AVAILABLE = False

class VolumeControl:
    def __init__(
        self,
        length_range: Tuple[float, float] = (50.0, 300.0),
        volume_range: Tuple[float, float] = (0.0, 1.0),
        smooth_factor: float = 0.2,
        notify_callback: Optional[Callable[[int], None]] = None,
        linux_amixer_device: Optional[str] = None
    ):
        """
        length_range: expected min,max distances from hand-tracker (pixels)
        volume_range: target volume scalar range [0.0-1.0] (float)
        smooth_factor: 0..1 (higher -> smoother / slower changes). 0 -> immediate.
        notify_callback: optional function called with new volume percent (0-100) after change.
        linux_amixer_device: optional amixer control name for Linux fallback (e.g., 'Master')
        """
        self.len_min, self.len_max = length_range
        self.vmin, self.vmax = volume_range
        self.smooth_factor = float(np.clip(smooth_factor, 0.0, 0.95))
        self.notify_callback = notify_callback
        self.linux_amixer_device = linux_amixer_device

        # internal state
        self._lock = threading.Lock()
        self.current_volume = None  # scalar 0.0-1.0

        # initialize pycaw if available
        self._pycaw = None
        if _PYCAW_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._pycaw = interface.QueryInterface(IAudioEndpointVolume)
                # try to read scalar volume if available
                try:
                    cur = self._pycaw.GetMasterVolumeLevelScalar()
                    self.current_volume = float(np.clip(cur, 0.0, 1.0))
                except Exception:
                    # if scalar not available, try dB read and map later
                    self.current_volume = 0.5
            except Exception:
                self._pycaw = None

        # fallback default
        if self.current_volume is None:
            self.current_volume = 0.5

    def _map_length_to_volume(self, length: float) -> float:
        """Map measured length to scalar volume in [vmin, vmax]"""
        vol = np.interp(length, [self.len_min, self.len_max], [self.vmin, self.vmax])
        vol = float(np.clip(vol, min(self.vmin, self.vmax), max(self.vmin, self.vmax)))
        return vol

    def _apply_volume(self, target: float):
        """Apply target scalar volume (0.0 - 1.0) with smoothing and platform calls."""
        with self._lock:
            if self.smooth_factor <= 0:
                new_v = target
            else:
                new_v = float(self.current_volume + (target - self.current_volume) * self.smooth_factor)
            new_v = float(np.clip(new_v, 0.0, 1.0))

            if abs(new_v - self.current_volume) < 1e-4:
                return

            # Try pycaw scalar first (Windows)
            if self._pycaw is not None:
                try:
                    # Some versions support SetMasterVolumeLevelScalar (0.0-1.0)
                    try:
                        self._pycaw.SetMasterVolumeLevelScalar(new_v, None)
                    except Exception:
                        # fallback: get dB range and set in dB (less common)
                        # Attempt to map scalar -> dB using volume range if possible
                        try:
                            volRange = self._pycaw.GetVolumeRange()  # (min, max, step) in dB
                            db_min, db_max = volRange[0], volRange[1]
                            db_target = np.interp(new_v, [0.0, 1.0], [db_min, db_max])
                            self._pycaw.SetMasterVolumeLevel(db_target, None)
                        except Exception:
                            # If even that fails, ignore but update internal state
                            pass
                    self.current_volume = new_v
                    if self.notify_callback:
                        try:
                            self.notify_callback(int(round(new_v * 100)))
                        except Exception:
                            pass
                    return
                except Exception:
                    # If pycaw call fails, continue to fallback
                    pass

            # Linux fallback using amixer (if available)
            if self.linux_amixer_device:
                try:
                    perc = int(round(new_v * 100))
                    import subprocess
                    subprocess.run(["amixer", "set", self.linux_amixer_device, f"{perc}%"], check=False)
                    self.current_volume = new_v
                    if self.notify_callback:
                        try:
                            self.notify_callback(int(round(new_v * 100)))
                        except Exception:
                            pass
                    return
                except Exception:
                    pass

            # Otherwise, we just update internal state (no-op on unsupported platform)
            self.current_volume = new_v
            if self.notify_callback:
                try:
                    self.notify_callback(int(round(new_v * 100)))
                except Exception:
                    pass

    def update_from_length(self, length: float):
        """
        Call this when your hand-tracker gives you the pixel distance between thumb and index.
        """
        target = self._map_length_to_volume(length)
        self._apply_volume(target)

    def update_from_landmarks(self, thumb: Tuple[float, float], index: Tuple[float, float]):
        """
        Call this when your tracker gives two (x,y) coordinates (in same coordinate system).
        thumb, index: (x, y)
        """
        dx = thumb[0] - index[0]
        dy = thumb[1] - index[1]
        length = float(np.hypot(dx, dy))
        self.update_from_length(length)

    def get_current_volume_percent(self) -> int:
        """Returns last-known volume as percent (0-100)."""
        with self._lock:
            return int(round(self.current_volume * 100))

    def set_direct_percent(self, percent: int):
        """Force-set volume by percent (0-100)."""
        p = int(np.clip(int(percent), 0, 100))
        self._apply_volume(p / 100.0)


# Example usage (no camera)
if __name__ == "__main__":
    def cb(v):
        print("Volume:", v, "%")

    ctrl = VolumeControl(length_range=(50, 300), volume_range=(0.0, 1.0), smooth_factor=0.2, notify_callback=cb, linux_amixer_device='Master')

    simulated = [60, 80, 130, 200, 280, 320, 200, 120, 60]
    for L in simulated:
        ctrl.update_from_length(L)
        time.sleep(0.2)
