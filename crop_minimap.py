import argparse
import os
import sys

import cv2
import numpy as np


def detect_minimap_panel(image):
    """
    Detect the full minimap HUD block in the bottom-right part of the screenshot.
    Returns (x, y, w, h).
    """
    h, w = image.shape[:2]

    x_start = int(w * 0.55)
    y_start = int(h * 0.50)
    roi = image[y_start:h, x_start:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 20000:
            continue

        ratio = ww / float(hh)
        if not (0.75 <= ratio <= 1.30):
            continue

        # Prefer larger candidates close to bottom-right
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        right_bias = cx / roi.shape[1]
        bottom_bias = cy / roi.shape[0]
        score = area * (0.5 + right_bias) * (0.5 + bottom_bias)

        if score > best_score:
            best_score = score
            best_box = (x + x_start, y + y_start, ww, hh)

    if best_box is None:
        raise RuntimeError("Minimap panel not found.")

    return best_box


def find_largest_square_contour(mask, min_area=8000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        ratio = w / float(h)
        if not (0.85 <= ratio <= 1.15):
            continue

        # Prefer large, square shapes
        squareness = 1.0 - abs(1.0 - ratio)
        score = area * squareness

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box


def refine_map_only(panel_img):
    """
    Detect the actual minimap square inside the minimap HUD block.
    Returns (x, y, w, h) relative to panel_img.
    """
    ph, pw = panel_img.shape[:2]

    # Ignore the portrait bar at the top
    top_skip = int(ph * 0.12)
    search = panel_img[top_skip:, :]

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

    # --- 1) Border color masks ---
    # Gold-ish border
    gold_mask = cv2.inRange(hsv, (10, 40, 60), (40, 255, 255))
    # Cyan / teal decorative lines sometimes present on the frame
    cyan_mask = cv2.inRange(hsv, (70, 30, 40), (120, 255, 255))

    color_mask = cv2.bitwise_or(gold_mask, cyan_mask)

    # Strengthen border lines
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_line, iterations=2)
    color_mask = cv2.dilate(color_mask, kernel_line, iterations=1)

    # --- 2) Dark square area mask ---
    # The minimap itself is mostly darker than the HUD around it
    dark_mask = cv2.inRange(gray, 0, 95)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_big, iterations=1)

    # --- 3) Combine both ---
    combined = cv2.bitwise_or(color_mask, dark_mask)

    # Close holes so the square becomes one large region
    kernel_combine = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_combine, iterations=3)

    box = find_largest_square_contour(combined, min_area=12000)

    # Fallback: if combined mask fails, rely only on dark square
    if box is None:
        box = find_largest_square_contour(dark_mask, min_area=12000)

    if box is None:
        raise RuntimeError("Exact minimap square not found.")

    x, y, w, h = box

    # Small outward padding to keep the full minimap border
    pad_x = max(2, int(w * 0.015))
    pad_y = max(2, int(h * 0.015))

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(search.shape[1] - x, w + 2 * pad_x)
    h = min(search.shape[0] - y, h + 2 * pad_y)

    return x, y + top_skip, w, h, color_mask, dark_mask, combined


def main():
    parser = argparse.ArgumentParser(description="Detect and crop ONLY the LoL minimap from a BMP screenshot.")
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

        mx, my, mw, mh, color_mask, dark_mask, combined = refine_map_only(panel)
        minimap_only = panel[my:my + mh, mx:mx + mw]

    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    if not cv2.imwrite(args.output, minimap_only):
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

        cv2.imwrite("debug_color_mask.bmp", color_mask)
        cv2.imwrite("debug_dark_mask.bmp", dark_mask)
        cv2.imwrite("debug_combined_mask.bmp", combined)

        print("Debug images saved:")
        print("- debug_detected_panel.bmp")
        print("- debug_panel_inner.bmp")
        print("- debug_color_mask.bmp")
        print("- debug_dark_mask.bmp")
        print("- debug_combined_mask.bmp")


if __name__ == "__main__":
    main()
