import argparse
import os
import sys
import cv2
import numpy as np


def find_game_area(image):
    """
    Trouve les bords du contenu jeu en détectant
    là où commence le fond gris macOS/OS.
    """
    h, w = image.shape[:2]

    def is_os_background(px):
        b, g, r = int(px[0]), int(px[1]), int(px[2])
        return (abs(b - g) < 15 and abs(g - r) < 15 and 30 < b < 80)

    # Bord droit : première colonne majoritairement grise (fond OS)
    game_right = w
    for x in range(w - 1, int(w * 0.5), -1):
        col = image[:, x]
        if sum(1 for px in col if is_os_background(px)) > h * 0.3:
            game_right = x + 1
            break

    # Bord bas : première rangée majoritairement grise
    game_bottom = h
    for y in range(h - 1, int(h * 0.5), -1):
        row = image[y, :]
        if sum(1 for px in row if is_os_background(px)) > w * 0.3:
            game_bottom = y + 1
            break

    return game_right, game_bottom


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap from a BMP screenshot.")
    parser.add_argument("--input",  required=True,         help="Path to input BMP")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP")
    parser.add_argument("--debug",  action="store_true",   help="Save debug overlay")
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

    game_right, game_bottom = find_game_area(image)
    print(f"Game area: {game_right}x{game_bottom}")

    # La minimap occupe ~26.5% de la largeur et ~36% de la hauteur du jeu,
    # toujours ancrée au coin bas-droit
    map_w = int(game_right * 0.265)
    map_h = int(game_bottom * 0.36)
    x1 = game_right - map_w
    y1 = game_bottom - map_h

    print(f"Minimap: ({x1},{y1}) → ({game_right},{game_bottom})")

    cropped = image[y1:game_bottom, x1:game_right]
    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x1, y1), (game_right, game_bottom), (0, 255, 0), 4)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
