import argparse
import os
import sys
import cv2


# Calibré sur 1920x1080 depuis les vrais BMP de jeu
# Minimap toujours ancrée en bas à droite
MINIMAP_LEFT   = 0.8729   # x=1676/1920
MINIMAP_TOP    = 0.5620   # y=607/1080
MINIMAP_RIGHT  = 0.9995   # x=1919/1920
MINIMAP_BOTTOM = 0.7630   # y=824/1080


def crop_minimap(image):
    h, w = image.shape[:2]
    x1 = int(w * MINIMAP_LEFT)
    y1 = int(h * MINIMAP_TOP)
    x2 = int(w * MINIMAP_RIGHT)
    y2 = int(h * MINIMAP_BOTTOM)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


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

    cropped, (x1, y1, x2, y2) = crop_minimap(image)
    print(f"Minimap: ({x1},{y1}) → ({x2},{y2}), size {x2-x1}x{y2-y1}")

    cv2.imwrite(args.output, cropped)
    print(f"Saved: {args.output}")

    if args.debug:
        dbg = image.copy()
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", dbg)
        print("Debug saved: debug_detected_minimap.bmp")


if __name__ == "__main__":
    main()
