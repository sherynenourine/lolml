import argparse
import os
import sys
import cv2

def crop_minimap_fixed(image):
    """
    Crop depuis le coin bas-droit, là où la minimap LoL est toujours ancrée.
    """
    h, w = image.shape[:2]

    # La minimap est carrée, ~20% de la largeur écran
    map_size = int(w * 0.205)

    # Ancrage bas-droit avec petite marge
    x = w - map_size - int(w * 0.002)
    y = h - map_size - int(h * 0.018)

    # Safety clamp
    x = max(0, x)
    y = max(0, y)
    if x + map_size > w:
        map_size = w - x
    if y + map_size > h:
        map_size = h - y

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

    x, y, w, h = crop_minimap_fixed(image)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Crop region: x={x}, y={y}, w={w}, h={h}")

    cropped = image[y:y + h, x:x + w]

    success = cv2.imwrite(args.output, cropped)
    if not success:
        print(f"Could not save output image: {args.output}")
        sys.exit(1)

    print(f"Minimap saved to: {args.output}")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite("debug_detected_minimap.bmp", debug_img)
        print("Debug image saved to: debug_detected_minimap.bmp")

if __name__ == "__main__":
    main()
