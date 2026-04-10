import argparse
import os
import sys
import cv2
import numpy as np


def is_border_color(pixel_bgr):
    """
    Vérifie si un pixel correspond à la couleur de bordure de la minimap.
    Marron/or : R élevé, G moyen, B faible
    """
    b, g, r = int(pixel_bgr[0]), int(pixel_bgr[1]), int(pixel_bgr[2])
    # Marron-or de la bordure LoL
    brown = (r > 80 and r < 180) and (g > 50 and g < 130) and (b < 80) and (r > g > b)
    # Vert-bleu foncé de la bordure
    teal = (b > 40 and b < 120) and (g > 50 and g < 110) and (r < 60) and (b >= r)
    return brown or teal


def scan_border(image):
    """
    Scanne depuis chaque bord pour trouver les limites exactes de la bordure minimap.
    On scanne uniquement dans le quart bas-droit.
    """
    h, w = image.shape[:2]

    # Zone de recherche restreinte au quart bas-droit
    min_x = int(w * 0.55)
    min_y = int(h * 0.50)

    # ── Bord DROIT : scan de droite à gauche ──────────────────────────
    right_x = None
    for col in range(w - 1, min_x, -1):
        column_pixels = image[min_y:h, col]
        matches = sum(1 for px in column_pixels if is_border_color(px))
        if matches > (h - min_y) * 0.15:  # 15% de la colonne = bordure trouvée
            right_x = col
            break

    # ── Bord BAS : scan de bas en haut ────────────────────────────────
    bottom_y = None
    for row in range(h - 1, min_y, -1):
        row_pixels = image[row, min_x:w]
        matches = sum(1 for px in row_pixels if is_border_color(px))
        if matches > (w - min_x) * 0.15:
            bottom_y = row
            break

    if right_x is None or bottom_y is None:
        return None

    # ── Bord GAUCHE : scan de gauche à droite ─────────────────────────
    left_x = None
    for col in range(min_x, right_x):
        column_pixels = image[min_y:bottom_y, col]
        matches = sum(1 for px in column_pixels if is_border_color(px))
        if matches > (bottom_y - min_y) * 0.15:
            left_x = col
            break

    # ── Bord HAUT : scan de haut en bas ───────────────────────────────
    top_y = None
    for row in range(min_y, bottom_y):
        row_pixels = image[row, min_x:right_x]
        matches = sum(1 for px in row_pixels if is_border_color(px))
        if matches > (right_x - min_x) * 0.15:
            top_y = row
            break

    if left_x is None or top_y is None:
        return None

    return left_x, top_y, right_x, bottom_y


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input",  required=True,         help="Path to input BMP")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP")
    parser.add_argument("--debug",  action="store_true",   help="Save debug image")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}"); sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Cannot read: {args.input}"); sys.exit(1)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    result = scan_border(image)

    if result is not None:
        left_x, top_y, right_x, bottom_y = result
        print(f"Border found: left={left_x}, top={top_y}, right={right_x}, bottom={bottom_y}")
        cropped = image[top_y:bottom_y + 1, left_x:right_x + 1]
    else:
        print("Detection failed → fallback coin bas-droit")
        h, w = image.shape[:2]
        size = int(w * 0.155)
        left_x = w - size - 5
        top_y  = h - size - 10
        cropped = image[top_y:top_y + size, left_x:left_x + size]
        right_x, bottom_y = left_x + size, top_y + size

    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
