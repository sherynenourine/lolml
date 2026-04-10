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

    # Focus on bottom-right area
    x_start = int(w * 0.50)
    y_start = int(h * 0.45)
    roi = image[y_start:h, x_start:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    roi_h, roi_w = roi.shape[:2]

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh

        if area < 12000:
            continue

        ratio = ww / float(hh)
        if not (0.70 <= ratio <= 1.35):
            continue

        # Favor large boxes near bottom-right
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        right_bias = cx / roi_w
        bottom_bias = cy / roi_h
        score = area * (1.0 + right_bias + bottom_bias)

        if score > best_score:
            best_score = score
            best_box = (x + x_start, y + y_start, ww, hh)

    if best_box is None:
        raise RuntimeError("Minimap panel not found.")

    return best_box


def refine_map_only(panel_img):
    """
    Crop the actual minimap square from inside the minimap panel.
    Returns (x, y, w, h) relative to panel_img.
    """
    ph, pw = panel_img.shape[:2]

    # Ignore known HUD areas around the map
    left_margin = int(pw * 0.04)
    right_margin = int(pw * 0.10)
    top_margin = int(ph * 0.16)
    bottom_margin = int(ph * 0.08)

    search = panel_img[top_margin:ph - bottom_margin, left_margin:pw - right_margin]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # The map area is mostly dark compared to the UI around it
    _, dark_mask = cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    sh, sw = search.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 8000:
            continue

        ratio = w / float(h)
        if not (0.82 <= ratio <= 1.18):
            continue

        # Favor boxes near center of the search zone
        cx = x + w / 2.0
        cy = y + h / 2.0
        center_score = 1.0 - (abs(cx - sw / 2) / sw + abs(cy - sh / 2) / sh) * 0.5

        score = area * max(center_score, 0.1)

        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    if best_box is None:
        # Fallback to fixed crop if contour detection fails
        x = int(pw * 0.07)
        y = int(ph * 0.18)
        w = int(pw * 0.78)
        h = int(ph * 0.78)
        size = min(w, h)
        return x, y, size, size, dark_mask

    x, y, w, h = best_box

    # Convert search coords back to panel coords
    x += left_margin
    y += top_margin

    # Small padding to keep full map border
    pad = max(2, int(min(w, h) * 0.02))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(pw - x, w + 2 * pad)
    h = min(ph - y, h + 2 * pad)

    # Force square
    size = min(w, h)
    return x, y, size, size, dark_mask


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

        mx, my, mw, mh, dark_mask = refine_map_only(panel)
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

        cv2.imwrite("debug_dark_mask.bmp", dark_mask)
        print("Debug images saved: debug_detected_panel.bmp, debug_panel_inner.bmp, debug_dark_mask.bmp")


if __name__ == "__main__":
    main()
