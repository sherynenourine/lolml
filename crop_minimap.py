import argparse
import os
import sys
from glob import glob

import cv2


def crop_minimap_fixed(image):
    """
    Fixed crop of the bottom-right minimap area.
    Returns (x, y, w, h).
    """
    h, w = image.shape[:2]

    crop_w = int(w * 0.18)
    crop_h = int(h * 0.24)

    x = int(w * 0.79)
    y = int(h * 0.67)

    if x + crop_w > w:
        crop_w = w - x
    if y + crop_h > h:
        crop_h = h - y

    return x, y, crop_w, crop_h


def process_one_image(input_path, output_path, save_debug=False):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Could not read image: {input_path}")
        return False

    x, y, w, h = crop_minimap_fixed(image)
    cropped = image[y:y + h, x:x + w]

    ok = cv2.imwrite(output_path, cropped)
    if not ok:
        print(f"Could not save output image: {output_path}")
        return False

    print(f"Saved: {output_path}")

    if save_debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        base_name = os.path.splitext(os.path.basename(output_path))[0]
        debug_path = os.path.join(os.path.dirname(output_path), f"{base_name}_debug.bmp")
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug: {debug_path}")

    return True


def process_directory(input_dir, output_dir, save_debug=False):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(input_dir, "*.bmp")))
    if not image_paths:
        print(f"No BMP images found in: {input_dir}")
        return False

    success_count = 0

    for input_path in image_paths:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        ok = process_one_image(input_path, output_path, save_debug=save_debug)
        if ok:
            success_count += 1

    print(f"Done: {success_count}/{len(image_paths)} images processed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Crop the LoL minimap area from BMP screenshots.")
    parser.add_argument("--input", help="Path to one input BMP image")
    parser.add_argument("--output", default="minimap.bmp", help="Path to one output BMP image")
    parser.add_argument("--input-dir", help="Folder containing BMP images")
    parser.add_argument("--output-dir", help="Folder where cropped BMP images will be saved")
    parser.add_argument("--debug", action="store_true", help="Save debug image(s) with crop box")
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}")
            sys.exit(1)

        ok = process_one_image(args.input, args.output, save_debug=args.debug)
        sys.exit(0 if ok else 1)

    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Input directory not found: {args.input_dir}")
            sys.exit(1)

        if not args.output_dir:
            print("You must provide --output-dir when using --input-dir")
            sys.exit(1)

        ok = process_directory(args.input_dir, args.output_dir, save_debug=args.debug)
        sys.exit(0 if ok else 1)

    else:
        print("You must provide either --input or --input-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
