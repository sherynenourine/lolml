import argparse
import os
import sys

import cv2


def detect_minimap(image):
    """
    Detect a minimap-like region in the bottom-right area of the screenshot.
    Returns (x, y, w, h).
    """
    h, w = image.shape[:2]

    # Search in a larger bottom-right zone
    x_start = int(w * 0.45)
    y_start = int(h * 0.45)
    roi = image[y_start:h, x_start:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Softer edge detection
    edges = cv2.Canny(gray, 30, 120)

    # Close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_score = -1

    roi_h, roi_w = roi.shape[:2]

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh

        # Lower minimum area
        if area < 8000:
            continue

        ratio = ww / float(hh)

        # Accept less perfectly square candidates
        if not (0.65 <= ratio <= 1.40):
            continue

        # Prefer big boxes near the bottom-right
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        right_bias = cx / roi_w
        bottom_bias = cy / roi_h

        score = area * (1.0 + 0.5 * right_bias + 0.5 * bottom_bias)

        if score > best_score:
            best_score = score
            best_box = (x + x_start, y + y_start, ww, hh)

    if best_box is None:
        raise RuntimeError("Minimap not found. Try another screenshot or adjust thresholds.")

    return best_box, edges


def main():
    parser = argparse.ArgumentParser(description="Detect and crop the minimap from a BMP screenshot.")
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
        (x, y, w, h), edges = detect_minimap(image)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    cropped = image[y:y + h, x:x + w]

    success = cv2.imwrite(args.output, cropped)
    if not success:
        print(f"Could not save output image: {args.output}")
        sys.exit(1)

    print(f"Minimap cropped and saved to: {args.output}")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", debug_img)
        cv2.imwrite("debug_edges.bmp", edges)
        print("Debug images saved: debug_detected_minimap.bmp, debug_edges.bmp")


if __name__ == "__main__":
    main()
