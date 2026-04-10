import argparse
import os
import sys
import cv2

def crop_minimap_fixed(image):
    """
    Fixed crop of the bottom-right minimap area.
    Returns (x, y, w, h).
    """
    h, w = image.shape[:2]

    # La minimap LoL est dans le coin bas-droite, SOUS les portraits champions
    crop_w = int(w * 0.185)
    crop_h = int(h * 0.185)
    x = int(w * 0.797)
    y = int(h * 0.765)  # Était 0.67 → trop haut, capturait les portraits

    # Safety clamp
    if x + crop_w > w:
        crop_w = w - x
    if y + crop_h > h:
        crop_h = h - y

    return x, y, crop_w, crop_h

def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap area from a BMP screenshot.")
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
    cropped = image[y:y + h, x:x + w]

    success = cv2.imwrite(args.output, cropped)
    if not success:
        print(f"Could not save output image: {args.output}")
        sys.exit(1)

    print(f"Minimap cropped to: {args.output} (region: x={x}, y={y}, w={w}, h={h})")

    if args.debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        debug_path = "debug_detected_minimap.bmp"
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved to: {debug_path}")

if __name__ == "__main__":
    main()
