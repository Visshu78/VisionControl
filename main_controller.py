"""
main_controller.py  (patched)

Fixes included:
- Mirror / handedness toggles
- Debug prints for distances, mouse/volume/brightness updates
- Option to invert length->value mapping
- Safer pyautogui/mapping logic
"""

import time
import math
import collections
import numpy as np
import cv2
import mediapipe as mp
import traceback
import pyautogui
# import your modular controllers (these files should be present)
from NewHandEye import MouseHandEyeController
from NewVol import VolumeControl
from NewBright import BrightnessControl

# -------------------------
# Settings / Hyperparams (tweak these)
# -------------------------
CAM_ID = 0

# open capture first, then query real size
cam = cv2.VideoCapture(CAM_ID)
# optional: request a preferred size (camera may ignore)
REQUEST_W, REQUEST_H = 520, 360
cam.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)

# small delay to let camera apply settings
time.sleep(0.2)

# query actual camera size (may differ from requested)
cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or REQUEST_W
cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or REQUEST_H

# screen size
screen_w, screen_h = pyautogui.size()

# protect against zero (rare) and use float ratios
if cam_w == 0 or cam_h == 0:
    cam_w, cam_h = REQUEST_W, REQUEST_H

frame_w_ratio = float(screen_w) / float(cam_w)
frame_h_ratio = float(screen_h) / float(cam_h)

print(f"Camera actual: {cam_w}x{cam_h}, Screen: {screen_w}x{screen_h}")
print(f"Mapping ratios -> w:{frame_w_ratio:.3f}, h:{frame_h_ratio:.3f}")

# -------------------------
# Length / smoothing defaults
# -------------------------
# These are starting points — tune them with your camera distance.
# If you use REQUEST_W/REQUEST_H = 520x360, LENGTH_MIN/MAX = 50..180 is reasonable.
LENGTH_MIN = 40.0
LENGTH_MAX = 180.0

# Recommended smoothing values for responsiveness:
# Lower = more responsive, Higher = smoother but slower.
VOLUME_SMOOTH = 0.01       # recommended: 0.0 - 0.10 for gestures
BRIGHTNESS_SMOOTH = 0.01   # same
# Mirror/frame toggles (important)
MIRROR_FRAME = True    # If True: flip frame horizontally BEFORE processing (recommended if webcam is mirrored)
SWAP_HANDS = False     # If True: swap left/right hand assignments (use if handedness feels reversed)
DISABLE_VISUALS = False  # set True to not show cv2 window

# If your distance->value feels reversed, flip this
INVERT_LENGTH_MAPPING = False  # False: larger distance -> higher value (default). True: invert.

# Finger detection indices
TIP_IDS = {"index": 8, "middle": 12, "ring": 16, "thumb": 4}
PIP_IDS = {"index": 6, "middle": 10, "ring": 14, "thumb_ip": 3}

# Face landmarks for simple eye metric
LEFT_EYE_UP = 159
LEFT_EYE_DOWN = 145
RIGHT_EYE_UP = 386
RIGHT_EYE_DOWN = 374

CALIBRATE_FRAMES = 25
CALIBRATE_WAIT_SECS = 1.0

CLICK_HOLD = 0.01      # 0.01 karna padega

# -------------------------
# Utilities
# -------------------------
def landmark_to_pixel(lm, frame_w, frame_h):
    return int(lm.x * frame_w), int(lm.y * frame_h)

def compute_eye_metric(face_landmarks, up_idx, down_idx):
    try:
        up = face_landmarks.landmark[up_idx]
        down = face_landmarks.landmark[down_idx]
        return abs(up.y - down.y)
    except Exception:
        return 0.0

def fingers_up_count(hand_landmarks):
    def is_up(tip_id, pip_id):
        try:
            return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y
        except Exception:
            return False
    fingers = {
        'index': is_up(TIP_IDS['index'], PIP_IDS['index']),
        'middle': is_up(TIP_IDS['middle'], PIP_IDS['middle']),
        'ring': is_up(TIP_IDS['ring'], PIP_IDS['ring'])
    }
    count = sum(1 for k in fingers if fingers[k])
    return count, fingers

def thumb_index_distance_px(hand_landmarks, frame_w, frame_h):
    try:
        tx, ty = landmark_to_pixel(hand_landmarks.landmark[TIP_IDS['thumb']], frame_w, frame_h)
        ix, iy = landmark_to_pixel(hand_landmarks.landmark[TIP_IDS['index']], frame_w, frame_h)
        return math.hypot(tx - ix, ty - iy)
    except Exception:
        return 0.0

# -------------------------
# Main
# -------------------------
def main():
    # Controllers
    mouse = MouseHandEyeController(smooth=0.20, click_threshold=CLICK_HOLD, double_click_threshold=0.18)
    volume = VolumeControl(length_range=(LENGTH_MIN, LENGTH_MAX), volume_range=(0.0,1.0),
                           smooth_factor=VOLUME_SMOOTH,
                           notify_callback=lambda v: print(f"[VOL] {v}%"))
    brightness = BrightnessControl(length_range=(LENGTH_MIN, LENGTH_MAX), brightness_range=(0,100),
                                   smooth_factor=BRIGHTNESS_SMOOTH,
                                   notify_callback=lambda b: print(f"[BRT] {b}%"))

    state = "MOUSE"
    last_state_change = time.time()

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.6, min_tracking_confidence=0.6)
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # Calibration for open-eye baseline
    print("Calibration: look at camera (open eyes). Starting in %.1f s..." % CALIBRATE_WAIT_SECS)
    time.sleep(CALIBRATE_WAIT_SECS)
    left_metrics, right_metrics = [], []
    captured = 0
    while captured < CALIBRATE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue
        if MIRROR_FRAME:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fres = face_mesh.process(rgb)
        if fres.multi_face_landmarks:
            fl = fres.multi_face_landmarks[0]
            lm_l = compute_eye_metric(fl, LEFT_EYE_UP, LEFT_EYE_DOWN)
            lm_r = compute_eye_metric(fl, RIGHT_EYE_UP, RIGHT_EYE_DOWN)
            if lm_l > 0 and lm_r > 0:
                left_metrics.append(lm_l); right_metrics.append(lm_r); captured += 1
        time.sleep(0.02)
    left_open = float(np.mean(left_metrics)) if left_metrics else 0.03
    right_open = float(np.mean(right_metrics)) if right_metrics else 0.03
    EYE_CLOSED_RATIO = 0.50
    left_thresh = left_open * EYE_CLOSED_RATIO
    right_thresh = right_open * EYE_CLOSED_RATIO
    print(f"[CALIB] left_open={left_open:.4f}, right_open={right_open:.4f}")
    print(f"[CALIB] left_thresh={left_thresh:.4f}, right_thresh={right_thresh:.4f}")

    left_q = collections.deque(maxlen=5)
    right_q = collections.deque(maxlen=5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)  # flip before processing so processing matches display

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_res = hands.process(frame_rgb)
            face_res = face_mesh.process(frame_rgb)

            left_hand = None
            right_hand = None

            # Use handedness if available; optionally swap
            if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
                for idx, hland in enumerate(hand_res.multi_hand_landmarks):
                    try:
                        label = hand_res.multi_handedness[idx].classification[0].label  # "Left"/"Right"
                    except Exception:
                        label = "Unknown"
                    if SWAP_HANDS:
                        # swap assignment if user prefers
                        label = "Left" if label == "Right" else ("Right" if label == "Left" else label)
                    if label == "Left":
                        left_hand = hland
                    elif label == "Right":
                        right_hand = hland

            # Fallback: if handedness not present, infer by x of landmark
            if (left_hand is None and right_hand is None) and hand_res.multi_hand_landmarks:
                for hland in hand_res.multi_hand_landmarks:
                    try:
                        idxx = hland.landmark[TIP_IDS['index']].x
                        if idxx < 0.5:
                            left_hand = hland
                        else:
                            right_hand = hland
                    except Exception:
                        pass

            # Mode switching via left hand (2->volume, 3->brightness)
            left_count = 0
            if left_hand is not None:
                left_count, left_fingers = fingers_up_count(left_hand)

            prev_state = state
            if left_count == 2:
                state = "VOLUME"
            elif left_count == 3:
                state = "BRIGHTNESS"
            else:
                state = "MOUSE"

            if state != prev_state:
                print(f"[STATE] {prev_state} -> {state}")
                last_state_change = time.time()
                if state == "MOUSE":
                    mouse.enable()
                else:
                    mouse.disable()

            # Volume / Brightness control using thumb-index distance from LEFT hand
            if left_hand is not None and state in ("VOLUME","BRIGHTNESS"):
                dist = thumb_index_distance_px(left_hand, FRAME_W, FRAME_H)
                if INVERT_LENGTH_MAPPING:
                    # invert mapping: small distance -> max, large -> min
                    dist = LENGTH_MIN + (LENGTH_MAX - (dist - LENGTH_MIN))
                # Print debug
                print(f"[DIST] left thumb-index: {dist:.1f}px (state={state})")
                if state == "VOLUME":
                    volume.update_from_length(dist)
                else:
                    brightness.update_from_length(dist)

            # Cursor control from RIGHT hand index when in MOUSE state
            if right_hand is not None and state == "MOUSE":
                try:
                    idx_lm = right_hand.landmark[TIP_IDS['index']]
                    x_norm = idx_lm.x
                    y_norm = idx_lm.y
                    # If MIRROR_FRAME=False but you still want mirrored display, you may need to flip x
                    # We already flipped the frame when MIRROR_FRAME=True, so coordinates align with display.
                    # Debug print (small)
                    # print(f"[MOUSE] norm coords: {x_norm:.3f}, {y_norm:.3f}")
                    mouse.move_cursor(x_norm, y_norm, normalized=True)
                except Exception as e:
                    print("Mouse move failed:", e)

            # Eye blink detection for clicks
            if face_res.multi_face_landmarks:
                fl = face_res.multi_face_landmarks[0]
                lm_l = compute_eye_metric(fl, LEFT_EYE_UP, LEFT_EYE_DOWN)
                lm_r = compute_eye_metric(fl, RIGHT_EYE_UP, RIGHT_EYE_DOWN)
                left_q.append(lm_l); right_q.append(lm_r)
                avg_l = float(np.mean(left_q))
                avg_r = float(np.mean(right_q))
                left_closed = avg_l < left_thresh
                right_closed = avg_r < right_thresh
                mouse.update_eyes(left_closed, right_closed)

            # Optional overlay (disable for headless)
            if not DISABLE_VISUALS:
                vis = frame.copy()
                cv2.putText(vis, f"State:{state}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if left_hand:
                    cv2.putText(vis, f"LCount:{left_count}", (12,56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
                try:
                    volp = volume.get_current_volume_percent()
                    brtp = brightness.get_current_brightness()
                except Exception:
                    volp = None; brtp = None
                if volp is not None:
                    cv2.putText(vis, f"Vol:{volp}%", (12,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,180,180), 2)
                if brtp is not None:
                    cv2.putText(vis, f"Brt:{brtp}%", (12,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)
                cv2.imshow("Controller (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print("Unhandled error:", e)
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()
        print("Exited")

if __name__ == "__main__":
    main()
