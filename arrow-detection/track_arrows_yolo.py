"""Detect FIRA HuroCup arrow markers with a YOLOv8s model trained on augmented photos.

Usage:
    python track_arrows_yolo.py                         # webcam 0, press q to quit
    python track_arrows_yolo.py --source samples/left.jpg --save out.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "weights" / "arrows_yolov8s.pt"
LABELS = {0: "Forward", 1: "Right", 2: "Left"}


def detect_arrows(model, frame, conf):
    """List of ((x1, y1, x2, y2), label, confidence)."""
    detections = []
    for result in model(frame, conf=conf, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(((x1, y1, x2, y2), LABELS.get(int(box.cls[0]), "Unknown"),
                               float(box.conf[0])))
    return detections


def draw(frame, detections, fps=None):
    for (x1, y1, x2, y2), label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
    return frame


def run_webcam(model, camera, conf):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous = time.perf_counter()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break
        detections = detect_arrows(model, frame, conf)
        now = time.perf_counter()
        fps, previous = 1.0 / max(now - previous, 1e-6), now
        cv2.imshow("Arrow detection - YOLOv8", draw(frame, detections, fps))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="0", help="webcam index or image path")
    ap.add_argument("--save", help="where to write the annotated image (image mode)")
    ap.add_argument("--conf", type=float, default=0.8, help="confidence threshold")
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    if args.source.isdigit():
        run_webcam(model, int(args.source), args.conf)
        return
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Cannot read {args.source}")
    detections = detect_arrows(model, frame, args.conf)
    print(f"{args.source}: {[f'{l} {c:.2f}' for _, l, c in detections] or 'no marker found'}")
    if args.save:
        cv2.imwrite(args.save, draw(frame, detections))


if __name__ == "__main__":
    main()
