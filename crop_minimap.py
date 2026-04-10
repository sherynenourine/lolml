import argparse
import os
import sys

import cv2
import numpy as np


def detect_minimap_panel(image):
    """
    Detect the full minimap HUD block in the bottom-right area.
    Returns (x, y, w, h).
    """
    h, w = image.shape[:2]

    x_start = int(w * 0.55)
    y_start = int(h * 0.55)
    roi = image[y_start:h, x_start:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh

        if area < 15000:
            continue

        ratio = ww / float(hh)
        if 0.75 <= ratio <= 1.25:
            if area > best_area:
                best_area = area
                best_box = (x + x_start, y + y_start, ww, hh)

    if best_box is None:
        raise RuntimeError("Minimap panel not found.")

    return best_box


def refine_map_only(panel_img):
    """
    Inside the minimap panel, detect the actual map content only.
    Returns (x, y, w, h) relative to panel_img.
    """
    ph, pw = panel_img.shape[:2]

    # Ignore the top band with champion portraits / HUD
    top_cut = int(ph * 0.18)
    search = panel_img[top_cut:ph, 0:pw]

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    # Keep pixels that are not too dark and somewhat saturated:
    # useful to keep the map interior and reject black frame/background
    mask = cv2.inRange(hsv, (0, 25, 25), (180, 255, 255))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 5000:
            continue

        ratio = w / float(h)
        if not (0.80 <= ratio <= 1.20):
            continue

        # Prefer large square-ish zones located in the lower-middle part
        cx = x + w / 2
        cy = y + h / 2
        center_bias = 1.0 - abs((cx / pw) - 0.5) * 0.5

        score = area * center_bias
        if score > best_score:
            best_score = score
            best_box = (x, y + top_cut, w, h)

    if best_box is None:
        raise RuntimeError("Map interior not found.")

    x, y, w, h = best_box

    # Small inward margin to remove the decorative border
    mx = max(3, int(w * 0.03))
    my = max(3, int(h * 0.03))

    x += mx
    y += my
    w -= 2 * mx
    h -= 2 * my

    if w <= 0 or h <= 0:
        raise RuntimeError("Refined crop is invalid.")

    return x, y, w, h


def main():
    parser = argparse.ArgumentParser(description="Detect and crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input", required=True, help="Path to input BMP image")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP image")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Could not read image: {args.input}")
        sys.exit(1)

    try:
        px, py, pw, ph = detect_minimap_panel(image)
        panel = image[py:py + ph, px:px + pw]

        mx, my, mw, mh = refine_map_only(panel)
        minimap_only = panel[my:my + mh, mx:mx + mw]

    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    success = cv2.imwrite(args.output, minimap_only)
    if not success:
        print(f"Could not save output image: {args.output}")
        sys.exit(1)

    print(f"Minimap-only crop saved to: {args.output}")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
        cv2.rectangle(debug_img, (px + mx, py + my), (px + mx + mw, py + my + mh), (255, 255, 255), 2)
        cv2.imwrite("debug_detected_panel.bmp", debug_img)

        debug_panel = panel.copy()
        cv2.rectangle(debug_panel, (mx, my), (mx + mw, my + mh), (255, 255, 255), 2)
        cv2.imwrite("debug_panel_inner.bmp", debug_panel)

        print("Debug images saved: debug_detected_panel.bmp, debug_panel_inner.bmp")


if __name__ == "__main__":
    main()
