import argparse
import os
import sys
import cv2
import numpy as np


def find_minimap(image):
    """
    Trouve la minimap en détectant la grande zone noire (fog of war)
    dans le quadrant bas-droit de l'écran.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels très sombres = intérieur de la minimap (fog of war)
    _, dark = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV)

    # Restreindre au quadrant bas-droit (la minimap est toujours là)
    mask = np.zeros_like(dark)
    mask[int(h * 0.25):, int(w * 0.25):] = 255
    dark = cv2.bitwise_and(dark, mask)

    # Remplir les trous (champions, icônes, etc.)
    k = np.ones((15, 15), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Trier par aire décroissante, prendre le plus grand contour quasi-carré
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / ch if ch > 0 else 0
        if 0.65 < ratio < 1.45 and cw > w * 0.08:
            # Ajouter la bordure (quelques pixels autour)
            pad = int(w * 0.005)
            x = max(0, x - pad)
            y = max(0, y - pad)
            cw = min(w - x, cw + 2 * pad)
            ch = min(h - y, ch + 2 * pad)
            return x, y, cw, ch

    return None


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input",  required=True,         help="Path to input BMP")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP")
    parser.add_argument("--debug",  action="store_true",   help="Save debug overlay image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Cannot read: {args.input}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Image: {w}x{h}")

    result = find_minimap(image)

    if result:
        x, y, cw, ch = result
        print(f"Minimap found: ({x},{y}) → ({x+cw},{y+ch}), size {cw}x{ch}")
        cropped = image[y:y+ch, x:x+cw]
    else:
        print("ERREUR: minimap non détectée.")
        sys.exit(1)

    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x, y), (x+cw, y+ch), (0, 255, 0), 4)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
