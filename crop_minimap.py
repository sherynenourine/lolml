import argparse
import os
import sys
import cv2
import numpy as np


def detect_minimap_border(image):
    """
    Détecte la bordure dorée de la minimap en scannant depuis le bas-droit.
    Retourne (x, y, w, h) de la minimap seule.
    """
    h, w = image.shape[:2]

    # Zone de recherche : quart bas-droit
    search_x = int(w * 0.6)
    search_y = int(h * 0.5)
    roi = image[search_y:h, search_x:w]

    # La bordure de la minimap est dorée/marron foncé → HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Plage couleur or/bronze de la bordure LoL
    lower_gold = np.array([10, 80, 80])
    upper_gold = np.array([35, 255, 200])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Morphologie pour nettoyer
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # La minimap est approximativement carrée
            ratio = cw / ch if ch > 0 else 0
            if 0.7 < ratio < 1.4 and area > 5000:
                best_area = area
                best = (x + search_x, y + search_y, cw, ch)

    return best


def crop_minimap_fallback(image):
    """
    Fallback : ancrage bas-droit, map uniquement (sans scoreboard portrait).
    """
    h, w = image.shape[:2]
    map_size = int(w * 0.155)   # taille carte seule (sans portraits)
    margin_x = int(w * 0.002)
    margin_y = int(h * 0.015)
    x = w - map_size - margin_x
    y = h - map_size - margin_y
    x = max(0, x)
    y = max(0, y)
    return x, y, map_size, map_size


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

    result = detect_minimap_border(image)
    if result is not None:
        x, y, w, h = result
        print(f"Minimap border detected: x={x}, y={y}, w={w}, h={h}")
    else:
        print("Border detection failed, using fallback.")
        x, y, w, h = crop_minimap_fallback(image)
        print(f"Fallback crop: x={x}, y={y}, w={w}, h={h}")

    cropped = image[y:y + h, x:x + w]
    success = cv2.imwrite(args.output, cropped)
    if not success:
        print(f"Could not save output: {args.output}")
        sys.exit(1)
    print(f"Minimap saved to: {args.output}")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", debug_img)
        print("Debug image saved to: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
