"""
detect.py — Real-time Driver Drowsiness Detection
Uses MediaPipe FaceMesh for landmark geometry + EyeCNN for eye state classification.

Usage:
    python detect.py                        # webcam (default)
    python detect.py --source video.mp4     # video file
    python detect.py --source 0             # explicit webcam index
    python detect.py --no-sound             # disable beep alerts
"""

import cv2
import numpy as np
import torch
import time
import argparse
from collections import deque
from scipy.spatial import distance as dist
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from model import EyeCNN

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ─── ARGS ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--source',     default='0',  help='Webcam index or video path')
parser.add_argument('--model',      default='driver_eye_model.pth')
parser.add_argument('--ear-thresh', type=float, default=0.25)
parser.add_argument('--mar-thresh', type=float, default=0.75)
parser.add_argument('--sleep-sec',  type=float, default=2.0)
parser.add_argument('--no-sound',   action='store_true')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOURCE = int(args.source) if args.source.isdigit() else args.source
# ─────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33,  160, 158, 133, 153, 144]
MOUTH = [61,  291, 39,  181, 0,   17,  269, 405]


# ─── LOAD MODEL ──────────────────────────────────────────────────
print(f'Device : {DEVICE}')
model = EyeCNN().to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model.eval()
print(f'✅ Model loaded  →  {args.model}')


# ─── LOAD FACE MESH ──────────────────────────────────────────────
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print('✅ FaceMesh ready\n')


# ─── MATH ────────────────────────────────────────────────────────
def get_ear(lms, pts):
    p = [lms[i] for i in pts]
    return (dist.euclidean(p[1], p[5]) + dist.euclidean(p[2], p[4])) / (2.0 * dist.euclidean(p[0], p[3]))

def get_mar(lms, pts):
    p = [lms[i] for i in pts]
    return (dist.euclidean(p[2], p[6]) + dist.euclidean(p[3], p[5])) / (2.0 * dist.euclidean(p[0], p[4]))

def preprocess_eye(crop):
    img = cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def beep(freq, dur):
    if not args.no_sound and WINSOUND_AVAILABLE:
        winsound.Beep(freq, dur)


# ─── DRAW HUD ────────────────────────────────────────────────────
def draw_hud(frame, status, color, ear, mar, yawn_count, fps, threshold_mar):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, 165), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.rectangle(frame, (10, 10), (440, 165), color, 2)
    cv2.putText(frame, status,
                (20, 52),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f'EAR: {ear:.2f}   MAR: {mar:.2f} (thresh: {threshold_mar:.2f})',
                (20, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Yawns (60s window): {yawn_count}',
                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.1f}',
                (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


# ─── CALIBRATION ─────────────────────────────────────────────────
cap = cv2.VideoCapture(SOURCE)

print("Calibrating — keep your mouth closed for 3 seconds...")
mar_samples  = []
cal_start    = time.time()

while time.time() - cal_start < 3.0:
    ret, frame = cap.read()
    if not ret:
        break
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for flm in results.multi_face_landmarks:
            h, w, _ = frame.shape
            lms = [(int(pt.x * w), int(pt.y * h)) for pt in flm.landmark]
            mar_samples.append(get_mar(lms, MOUTH))

    remaining = int(3.0 - (time.time() - cal_start)) + 1
    cv2.putText(frame, f"Calibrating — keep mouth closed ({remaining}s)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Driver Safety System", frame)
    cv2.waitKey(1)

if mar_samples:
    baseline_mar  = sum(mar_samples) / len(mar_samples)
    YAWN_THRESHOLD = baseline_mar - 0.12
    print(f"Baseline MAR : {baseline_mar:.3f}")
    print(f"Yawn threshold set to : {YAWN_THRESHOLD:.3f}\n")
else:
    YAWN_THRESHOLD = args.mar_thresh
    print(f"No face detected during calibration — using default threshold: {YAWN_THRESHOLD}\n")


# ─── MAIN LOOP ───────────────────────────────────────────────────
sleep_timer = None
yawn_log    = deque()
yawn_active = False
yawn_start  = None
fps_log     = deque(maxlen=30)
prev_time   = time.time()

print("▶  Running — press Q to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    fps_log.append(1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    fps = sum(fps_log) / len(fps_log)

    h, w, _       = frame.shape
    rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results        = face_mesh.process(rgb)
    status, color  = 'Monitoring', (0, 200, 0)
    ear = mar = 0.0

    if results.multi_face_landmarks:
        for flm in results.multi_face_landmarks:
            lms = [(int(pt.x * w), int(pt.y * h)) for pt in flm.landmark]

            # ── 1. Geometry ───────────────────────────────────────
            ear = (get_ear(lms, L_EYE) + get_ear(lms, R_EYE)) / 2.0
            mar =  get_mar(lms, MOUTH)

            # ── 2. CNN eye check ──────────────────────────────────
            is_closed = False
            if ear < args.ear_thresh:
                ex = lms[R_EYE[0]][0];  ey = lms[R_EYE[1]][1]
                ew = lms[R_EYE[3]][0] - ex
                eh = lms[R_EYE[4]][1] - ey
                x1, x2 = max(ex-10, 0), min(ex+ew+10, w)
                y1, y2 = max(ey-10, 0), min(ey+eh+10, h)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    with torch.no_grad():
                        conf = model(preprocess_eye(crop)).item()
                    if conf < 0.5:
                        is_closed = True

            # ── 3. Sleep alert ────────────────────────────────────
            if is_closed:
                if sleep_timer is None:
                    sleep_timer = time.time()
                elif time.time() - sleep_timer > args.sleep_sec:
                    status, color = 'SLEEP ALERT!', (0, 0, 255)
                    beep(1000, 200)
            else:
                sleep_timer = None

            # ── 4. Yawn detection (calibrated + duration-gated) ───
            if mar < YAWN_THRESHOLD:
                if yawn_start is None:
                    yawn_start = time.time()
                elif time.time() - yawn_start > 0.65:   # 0.65s sustained = yawn
                    if not yawn_active:
                        yawn_log.append(time.time())
                        yawn_active = True
            else:
                if yawn_active:
                    yawn_active = False   # ready to count next yawn
                yawn_start = None

            # Drop yawns outside 60s window
            while yawn_log and time.time() - yawn_log[0] > 60.0:
                yawn_log.popleft()

            if len(yawn_log) >= 3:
                status, color = 'FATIGUE ALERT!', (0, 140, 255)
                beep(500, 100)

    draw_hud(frame, status, color, ear, mar, len(yawn_log), fps, YAWN_THRESHOLD)
    cv2.imshow('Driver Safety System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print('System stopped.')