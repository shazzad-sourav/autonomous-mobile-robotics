"""Detect FIRA HuroCup arrow markers (left / right / forward) with classical OpenCV.

Pipeline: grayscale -> Gaussian blur -> Canny edges -> contours -> keep
4-sided polygons (the marker's square border) -> classify the arrow inside
by where the centroid of its largest contour falls (left / middle / right third).

Usage:
    python track_arrows_cv.py                         # webcam 0, press q to quit
    python track_arrows_cv.py --source samples/left.jpg --save out.jpg
"""

import argparse
import time

import cv2

MIN_AREA = 300      # px, smaller contours are noise
BORDER = 15         # px trimmed from the marker box before classifying


def classify_arrow(roi):
    """Return 'Left', 'Right', 'Forward' or 'Unknown' for a cropped marker.

    The centroid rule was drafted with ChatGPT (prompt: "Generate a python
    function that classifies arrow direction (left, right, forward) using
    opencv", 14 Feb 2025) and adapted by the team.
    """
    if roi.size == 0:
        return "Unknown"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"
    moments = cv2.moments(max(contours, key=cv2.contourArea))
    if moments["m00"] == 0:
        return "Unknown"
    cx = moments["m10"] / moments["m00"]
    width = roi.shape[1]
    if cx < width / 3:
        return "Left"
    if cx > 2 * width / 3:
        return "Right"
    return "Forward"


def detect_arrows(frame):
    """List of ((x, y, w, h), label) for every marker found in a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 200, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        roi = frame[y + BORDER:y + h - BORDER, x + BORDER:x + w - BORDER]
        detections.append(((x, y, w, h), classify_arrow(roi)))
    return detections


def draw(frame, detections, fps=None):
    for (x, y, w, h), label in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, label, (x, max(y - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
    return frame


def run_webcam(camera):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break
        detections = detect_arrows(frame)
        now = time.perf_counter()
        fps, previous = 1.0 / max(now - previous, 1e-6), now
        cv2.imshow("Arrow detection - CV", draw(frame, detections, fps))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="0", help="webcam index or image path")
    ap.add_argument("--save", help="where to write the annotated image (image mode)")
    args = ap.parse_args()

    if args.source.isdigit():
        run_webcam(int(args.source))
        return
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Cannot read {args.source}")
    # Thresholds are tuned for the 640 x 480 webcam stream
    frame = cv2.resize(frame, (640, 480))
    detections = detect_arrows(frame)
    print(f"{args.source}: {[label for _, label in detections] or 'no marker found'}")
    if args.save:
        cv2.imwrite(args.save, draw(frame, detections))


if __name__ == "__main__":
    main()
