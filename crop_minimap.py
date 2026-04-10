import argparse
import os
import sys
import cv2
import numpy as np


def is_teal_frame(px):
    """
    Détecte la couleur du cadre teal/vert-foncé de la minimap LoL.
    BGR ~(42, 55, 53) — tous les canaux proches, sombres, légèrement verts.
    """
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return (30 < b < 80 and 40 < g < 80 and 30 < r < 80
            and abs(b - g) < 25 and abs(g - r) < 25)


def find_minimap(image):
    h, w = image.shape[:2]

    # ── Zone de recherche : moitié droite + moitié basse ──────────────
    sx = int(w * 0.55)
    sy = int(h * 0.40)

    # ── LEFT : première colonne avec beaucoup de pixels teal ──────────
    left_x = None
    for x in range(sx, int(w * 0.85)):
        col = image[sy:h, x]
        if sum(1 for px in col if is_teal_frame(px)) > int((h - sy) * 0.25):
            left_x = x
            break

    # ── RIGHT : dernière colonne avec beaucoup de pixels teal ─────────
    right_x = None
    for x in range(w - 1, sx, -1):
        col = image[sy:h, x]
        if sum(1 for px in col if is_teal_frame(px)) > int((h - sy) * 0.25):
            right_x = x
            break

    if left_x is None or right_x is None:
        return None

    map_width = right_x - left_x

    # ── TOP : première ligne avec beaucoup de pixels teal ─────────────
    top_y = None
    for y in range(sy, int(h * 0.85)):
        row = image[y, left_x:right_x]
        if sum(1 for px in row if is_teal_frame(px)) > map_width * 0.55:
            top_y = y
            break

    if top_y is None:
        return None

    # La minimap est quasi-carrée → bottom = top + largeur
    bottom_y = min(top_y + map_width, h - 5)

    return left_x, top_y, right_x, bottom_y


def crop_minimap_fallback(image):
    """Fallback ancré bas-droit si la détection échoue."""
    h, w = image.shape[:2]
    size = int(w * 0.195)
    x = w - size - 5
    y = h - size - 5
    return max(0, x), max(0, y), w - 5, h - 5


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
        left_x, top_y, right_x, bottom_y = result
        print(f"Minimap detected: ({left_x},{top_y}) → ({right_x},{bottom_y})")
    else:
        print("Detection failed → fallback")
        left_x, top_y, right_x, bottom_y = crop_minimap_fallback(image)

    cropped = image[top_y:bottom_y, left_x:right_x]
    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 4)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
