import argparse
import os
import sys
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True,         help="Path to input BMP")
    parser.add_argument("--output",  default="minimap.bmp", help="Path to output BMP")
    parser.add_argument("--size",    type=float, default=0.21, help="Fraction de la largeur")
    parser.add_argument("--pad-left",type=int,   default=20,   help="Pixels supplémentaires à gauche")
    parser.add_argument("--pad-top", type=int,   default=15,   help="Pixels supplémentaires en haut")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Cannot read: {args.input}")
        sys.exit(1)

    h, w = image.shape[:2]
    size = int(min(w, h) * args.size)

    x1 = w - size - args.pad_left
    y1 = h - size - args.pad_top

    cropped = image[y1:h, x1:w]
    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output} ({w-x1}x{h-y1})")


if __name__ == "__main__":
    main()
