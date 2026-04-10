import argparse
import os
import sys
import cv2
import numpy as np


def find_minimap_rect(image):
    """
    Cherche le grand rectangle de la minimap dans le quart bas-droit de l'écran.
    La minimap LoL est toujours le plus grand rectangle dans cette zone.
    """
    h, w = image.shape[:2]

    # On cherche uniquement dans le quart bas-droit
    sx = int(w * 0.65)
    sy = int(h * 0.55)
    roi = image[sy:h, sx:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        # Approximer en polygone
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / ch if ch > 0 else 0

        # La minimap est quasi-carrée et grande
        if 0.75 < ratio < 1.25 and area > best_area:
            best_area = area
            best = (x + sx, y + sy, cw, ch)

    return best


def crop_minimap_fallback(image):
    """
    Fallback ancré au coin bas-droit, taille minimap seule.
    """
    h, w = image.shape[:2]
    # ~20% largeur pour la zone UI totale, mais la carte = ~77% de cette zone
    ui_w = int(w * 0.205)
    map_size = int(ui_w * 0.92)
    x = w - map_size - int(w * 0.003)
    y = h - map_size - int(h * 0.015)
    return max(0, x), max(0, y), map_size, map_size


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input", required=True, help="Path to input BMP image")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP image")
    parser.add_argument("--debug", action="store_true", help="Save a debug image with crop box")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Could not read image: {args.input}")
        sys.exit(1)

    result = find_minimap_rect(image)

    if result is not None:
        x, y, cw, ch = result
        # Forcer un carré (prendre le plus petit côté)
        size = min(cw, ch)
        # Réancrer au coin bas-droit du rect trouvé
        x = x + cw - size
        y = y + ch - size
        print(f"Detected minimap: x={x}, y={y}, size={size}")
    else:
        print("Detection failed → fallback")
        x, y, size, _ = crop_minimap_fallback(image)
        print(f"Fallback: x={x}, y={y}, size={size}")

    cropped = image[y:y + size, x:x + size]
    cv2.imwrite(args.output, cropped)
    print(f"Saved to: {args.output}")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + size, y + size), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", debug_img)
        print("Debug saved to: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
