# 🚗 Driver Drowsiness Detection System

> Real-time fatigue monitoring using a dual-pipeline approach — facial landmark geometry (EAR/MAR) combined with a CNN eye classifier — running on GPU via PyTorch + CUDA.

---

## How It Works

```
Webcam Frame
     │
     ▼
MediaPipe FaceMesh ──► 468 facial landmarks
     │
     ├──► EAR (Eye Aspect Ratio) ──► below threshold?
     │         │
     │         ▼
     │    EyeCNN classifier ──► confirms open/closed
     │         │
     │         ▼
     │    closed > 2s ──► 🔴 SLEEP ALERT
     │
     └──► MAR (Mouth Aspect Ratio) ──► yawn detected?
               │
               ▼
          3+ yawns / 60s ──► 🟠 FATIGUE ALERT
```

### Two-stage eye detection (why this matters)
Pure EAR geometry gives false positives — blinking, lighting changes, glasses glare. The CNN second stage confirms the eye state from the actual pixel crop, making alerts far more reliable.

### Alert System
| Condition | Alert |
|---|---|
| Eyes closed > 2 seconds | 🔴 SLEEP ALERT + 1000Hz beep |
| 3+ yawns within 60 seconds | 🟠 FATIGUE ALERT + 500Hz beep |

---

## Project Structure

```
driver-drowsiness-detection/
├── model.py          # EyeCNN architecture (shared)
├── train.py          # Training pipeline with curves & early stopping
├── detect.py         # Real-time webcam/video detection
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone
```bash
git clone https://github.com/aadidevvs2007/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### 2. Activate environment
```bash
conda activate D:\envs\project_env
```

### 3. Install dependencies
```bash
pip install mediapipe==0.10.14 protobuf==4.25.3 scipy==1.18.0
```
> `torch`, `torchvision`, `opencv-python`, `numpy`, `matplotlib` assumed pre-installed.

### 4. Get the dataset
Download **Drowsiness Detection** from Kaggle:
👉 https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection

Extract so structure is:
```
D:\Projects\DriverDrowsinessDetection\
  closed_eye\
  open_eye\
```

---

## Usage

### Train
```bash
python train.py

# Custom options
python train.py --data D:/path/to/dataset --epochs 20 --batch 64
```
- Auto-splits 70% train / 15% val / 15% test
- Saves best checkpoint as `driver_eye_model.pth`
- Saves `training_curves.png` after training

### Detect
```bash
# Webcam
python detect.py

# Video file
python detect.py --source path/to/video.mp4

# Tune thresholds
python detect.py --ear-thresh 0.22 --sleep-sec 1.5

# No sound (Linux/Mac or testing)
python detect.py --no-sound
```

---

## Model Architecture

```
Input (3 × 64 × 64)
  │
  ├── Conv2d(3→32)  + BN + ReLU + MaxPool   →  32 × 32 × 32
  ├── Conv2d(32→64) + BN + ReLU + MaxPool   →  64 × 16 × 16
  ├── Conv2d(64→128)+ BN + ReLU + MaxPool   →  128 × 8 × 8
  │
  ├── Flatten  →  8192
  ├── Linear(8192→256) + ReLU + Dropout(0.4)
  └── Linear(256→1) + Sigmoid
```

---

## Requirements

- Python 3.12
- CUDA-capable GPU (CPU fallback supported)
- Webcam or video file
- Windows (`winsound` for audio alerts; use `--no-sound` on Linux/Mac)

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| PyTorch | 2.11.0+cu128 | CNN training & inference |
| MediaPipe | 0.10.14 | Face mesh & landmark detection |
| OpenCV | 5.0.0.93 | Frame capture & display |
| SciPy | 1.18.0 | EAR/MAR distance computation |
| Matplotlib | 3.11.0 | Training curve visualization |

---

## Author

**Aadidev** — B.Tech AI/DS, Amrita Coimbatore

