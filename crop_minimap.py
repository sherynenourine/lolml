import argparse
import os
import sys

import cv2


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
    Crop only the inner minimap area from the detected minimap HUD block.
    Uses fixed relative margins inside the panel for better stability.
    Returns (x, y, w, h) relative to panel_img.
    """
    ph, pw = panel_img.shape[:2]

    left = int(pw * 0.11)
    right = int(pw * 0.11)
    top = int(ph * 0.20)
    bottom = int(ph * 0.10)

    x = left
    y = top
    w = pw - left - right
    h = ph - top - bottom

    size = min(w, h)
    x = x + (w - size) // 2
    y = y + (h - size) // 2

    return x, y, size, size


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
