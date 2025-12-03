## Project Overview

VisionControl is a computer vision-based gesture control system that uses hand tracking and facial recognition to control the mouse cursor, system volume, and screen brightness. Built with OpenCV, MediaPipe, and PyAutoGUI.

## Environment Setup

**Python Version**: Python 3.x with virtual environment

**Install dependencies**:
```powershell
pip install -r requirements.txt
```

**Key dependencies**:
- `opencv-python` - Camera capture and video processing
- `mediapipe` - Hand and face landmark detection
- `pyautogui` - Mouse control
- `pycaw` - Windows audio control
- `screen-brightness-control` - Display brightness control
- `pywin32`, `comtypes` - Windows API access

## Running the Application

**Start the main controller**:
```powershell
python main_controller.py
```

**Test individual modules**:
```powershell
python handlTrackingModule.py    # Test hand tracking only
python NewHandEye.py              # Test mouse controller
python NewVol.py                  # Test volume controller
python NewBright.py               # Test brightness controller
```

**Exit application**: Press `q` in the OpenCV window or `Ctrl+C` in terminal

## Architecture

### Control Flow

The system operates in three modes, switched via **left hand** finger count:
- **MOUSE mode** (default/0-1 fingers): Right hand index finger controls cursor, eye blinks trigger clicks
- **VOLUME mode** (2 fingers): Left hand thumb-index distance controls system volume
- **BRIGHTNESS mode** (3 fingers): Left hand thumb-index distance controls screen brightness

### Core Components

**main_controller.py**
- Main control loop integrating camera, MediaPipe, and all controllers
- Handles mode switching based on left hand finger detection
- Performs calibration on startup for eye-blink detection
- Manages frame processing, landmark detection, and visual overlay

**NewHandEye.py** (`MouseHandEyeController`)
- Maps hand position to screen coordinates with smoothing
- Implements eye-blink click detection with timing thresholds
- Supports left click (both eyes), right click (one eye), and double-click
- Can be enabled/disabled based on current mode

**NewVol.py** (`VolumeControl`)
- Maps finger distance to volume level (0.0-1.0 scalar)
- Uses pycaw for Windows audio control, with Linux amixer fallback
- Applies exponential smoothing to prevent jitter
- Thread-safe implementation

**NewBright.py** (`BrightnessControl`)
- Maps finger distance to brightness percentage (0-100)
- Uses `screen-brightness-control` library
- Smoothing and thread-safety similar to volume control

**handlTrackingModule.py** (`handDetector`)
- Reusable MediaPipe Hands wrapper
- Provides landmark detection and drawing utilities
- Used as reference but not imported by main controller

### Key Landmarks

**Hand landmarks** (MediaPipe indices):
- Thumb tip: 4, Index tip: 8, Middle tip: 12, Ring tip: 16
- Used for finger counting and distance measurements

**Face landmarks** (MediaPipe FaceMesh indices):
- Left eye: up=159, down=145
- Right eye: up=386, down=374
- Eye aspect ratio used for blink detection

### Calibration

On startup, the system captures 25 frames to establish baseline eye-open metrics. The `EYE_CLOSED_RATIO` (default 0.40 ) determines sensitivity—lower values make blink detection more sensitive.

### Configuration Constants

Key tunable parameters in `main_controller.py`:
- `LENGTH_MIN/MAX` (20-120px): Finger distance range for volume/brightness
- `VOLUME_SMOOTH/BRIGHTNESS_SMOOTH` (0.15): Smoothing factor
- `CLICK_HOLD` (0.01s): Minimum eye-closed duration for click
- `MIRROR_FRAME` (True): Flip camera for natural interaction
- `CAM_ID` (0): Camera device index

## Development Notes

### Windows-Specific

This project is designed for Windows due to:
- `pycaw` and `pywin32` for audio control (Windows-only)
- `screen-brightness-control` has best support on Windows
- Use PowerShell commands, not bash

### Threading & Smoothing

All controllers use thread locks for state management and exponential smoothing to prevent jittery control. The smoothing factor determines responsiveness vs. stability trade-off.

### Frame Processing Pipeline

1. Capture frame → Mirror if enabled
2. Convert BGR to RGB for MediaPipe
3. Process hands and face in parallel
4. Determine mode from left hand finger count
5. Execute mode-specific control (cursor/volume/brightness)
6. Render visual overlay and display

### Coordinate Systems

- **MediaPipe**: Normalized coordinates (0.0-1.0)
- **Camera**: Pixel coordinates based on actual frame size
- **Screen**: Absolute pixel coordinates from pyautogui
- Controllers handle mapping between these systems

## Common Issues

**Camera not opening**: Check `CAM_ID` matches your camera index, try 0, 1, or 2

**Clicks too sensitive/insensitive**: Adjust `EYE_CLOSED_RATIO` in calibration section or `CLICK_HOLD` timing

**Volume/brightness not responding**: Check distance is within `LENGTH_MIN/MAX` range, verify system permissions

**Jittery control**: Increase smoothing factors (`VOLUME_SMOOTH`, `BRIGHTNESS_SMOOTH`, mouse `smooth` parameter)
