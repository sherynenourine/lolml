import argparse
import os
import sys
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,         help="Path to input BMP")
    parser.add_argument("--output", default="minimap.bmp", help="Path to output BMP")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Cannot read: {args.input}")
        sys.exit(1)

    h, w = image.shape[:2]
    size = int(min(w, h) * 0.21)

    x1 = w - size - 90
    y1 = h - size - 80

    cropped = image[y1:h, x1:w]
    cv2.imwrite(args.output, cropped)
    print(f"Image: {w}x{h} → crop: ({x1},{y1}) taille {w-x1}x{h-y1}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
