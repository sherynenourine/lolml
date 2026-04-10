import argparse
import os
import sys
import cv2
import numpy as np


def find_minimap_by_border_colors(image, debug=False):
    """
    Détecte la bordure de la minimap via ses couleurs caractéristiques :
    - Bande marron/or  (BGR ~  40, 85, 122)
    - Bande vert foncé (BGR ~ 55, 65,  25)
    """
    h, w = image.shape[:2]

    # ── Masque couleur marron/or de la bordure ──────────────────────────
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Marron-or : H 15-35, S 80-220, V 60-170
    mask_brown = cv2.inRange(hsv,
                             np.array([15,  80,  60]),
                             np.array([35, 220, 170]))

    # Vert-bleu foncé de la bordure : H 85-105, S 60-200, V 30-100
    mask_teal = cv2.inRange(hsv,
                            np.array([85,  60,  30]),
                            np.array([105, 200, 100]))

    # Combinaison des deux bandes
    mask = cv2.bitwise_or(mask_brown, mask_teal)

    # Garder uniquement le quart bas-droit (la minimap est toujours là)
    region_mask = np.zeros_like(mask)
    region_mask[int(h * 0.5):, int(w * 0.55):] = 255
    mask = cv2.bitwise_and(mask, region_mask)

    # Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

    if debug:
        cv2.imwrite("debug_mask.bmp", mask)

    # ── Trouver le bord DROIT de la bordure ─────────────────────────────
    # Projection horizontale : colonne la plus à droite avec pixels allumés
    col_sums = np.sum(mask, axis=0)
    right_cols = np.where(col_sums > 50)[0]
    if len(right_cols) == 0:
        return None
    right_x = int(right_cols.max())

    # ── Trouver le bord BAS de la bordure ───────────────────────────────
    row_sums = np.sum(mask, axis=1)
    bottom_rows = np.where(row_sums > 50)[0]
    if len(bottom_rows) == 0:
        return None
    bottom_y = int(bottom_rows.max())

    # ── Trouver le bord GAUCHE et HAUT ──────────────────────────────────
    left_cols = np.where(col_sums > 50)[0]
    left_x = int(left_cols.min())

    top_rows = np.where(row_sums > 50)[0]
    top_y = int(top_rows.min())

    map_w = right_x - left_x
    map_h = bottom_y - top_y

    # Forcer un carré (on prend le min des deux dimensions)
    size = min(map_w, map_h)

    # Réancrer : on part du coin bas-droit détecté
    x = right_x - size
    y = bottom_y - size

    return max(0, x), max(0, y), size, size


def crop_minimap_fallback(image):
    h, w = image.shape[:2]
    size = int(w * 0.155)
    x = w - size - int(w * 0.003)
    y = h - size - int(h * 0.015)
    return max(0, x), max(0, y), size, size


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input",  required=True,          help="Path to input BMP")
    parser.add_argument("--output", default="minimap.bmp",  help="Path to output BMP")
    parser.add_argument("--debug",  action="store_true",    help="Save debug images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Could not read image: {args.input}")
        sys.exit(1)

    result = find_minimap_by_border_colors(image, debug=args.debug)

    if result is not None:
        x, y, size, _ = result
        print(f"Border detected → x={x}, y={y}, size={size}")
    else:
        print("Detection failed → fallback (coin bas-droit)")
        x, y, size, _ = crop_minimap_fallback(image)
        print(f"Fallback → x={x}, y={y}, size={size}")

    cropped = image[y:y + size, x:x + size]
    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x, y), (x + size, y + size), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug overlay saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
