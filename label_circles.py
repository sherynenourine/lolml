import argparse
import json
import math
import os
import sys
from glob import glob

import cv2


WINDOW_NAME = "Label circles"


class CircleLabeler:
    def __init__(self, image_paths, output_dir):
        self.image_paths = image_paths
        self.output_dir = output_dir

        self.index = 0
        self.image = None
        self.display_image = None
        self.current_path = None

        self.circles = []
        self.is_drawing = False
        self.start_point = None
        self.current_point = None

    def load_current_image(self):
        if self.index >= len(self.image_paths):
            return False

        self.current_path = self.image_paths[self.index]
        self.image = cv2.imread(self.current_path)

        if self.image is None:
            raise RuntimeError(f"Could not read image: {self.current_path}")

        self.circles = []
        self.is_drawing = False
        self.start_point = None
        self.current_point = None
        self.refresh_display()
        return True

    def refresh_display(self):
        self.display_image = self.image.copy()

        # Draw saved circles
        for i, circle in enumerate(self.circles, start=1):
            x = circle["center_x"]
            y = circle["center_y"]
            r = circle["radius"]

            cv2.circle(self.display_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(self.display_image, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(
                self.display_image,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw current preview circle while dragging
        if self.is_drawing and self.start_point and self.current_point:
            cx, cy = self.start_point
            px, py = self.current_point
            radius = int(round(math.hypot(px - cx, py - cy)))
            cv2.circle(self.display_image, (cx, cy), radius, (0, 255, 255), 2)
            cv2.circle(self.display_image, (cx, cy), 2, (0, 255, 255), -1)

        # Status text
        info_1 = f"Image {self.index + 1}/{len(self.image_paths)}"
        info_2 = os.path.basename(self.current_path)
        info_3 = "Left drag: add circle | Right click: undo | Space: save+next | Esc: quit"

        cv2.putText(
            self.display_image,
            info_1,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            self.display_image,
            info_2,
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            self.display_image,
            info_3,
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def save_current_labels(self):
        os.makedirs(self.output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.current_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.json")

        payload = {
            "image_filename": os.path.basename(self.current_path),
            "image_path": self.current_path,
            "image_width": int(self.image.shape[1]),
            "image_height": int(self.image.shape[0]),
            "num_circles": len(self.circles),
            "circles": self.circles,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved labels: {output_path}")

    def go_to_next_image(self):
        self.index += 1
        return self.load_current_image()

    def add_circle_from_drag(self):
        if not self.start_point or not self.current_point:
            return

        cx, cy = self.start_point
        px, py = self.current_point
        radius = int(round(math.hypot(px - cx, py - cy)))

        if radius < 3:
            radius = 10

        self.circles.append(
            {
                "center_x": int(cx),
                "center_y": int(cy),
                "radius": int(radius),
            }
        )

    def undo_last_circle(self):
        if self.circles:
            removed = self.circles.pop()
            print(f"Removed circle: {removed}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.refresh_display()

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self.current_point = (x, y)
            self.refresh_display()

        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.current_point = (x, y)
            self.add_circle_from_drag()
            self.is_drawing = False
            self.start_point = None
            self.current_point = None
            self.refresh_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo_last_circle()
            self.refresh_display()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label circles on BMP images to create a dataset."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing BMP images",
    )
    parser.add_argument(
        "--output-dir",
        default="labels",
        help="Folder where JSON label files will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    image_paths = sorted(glob(os.path.join(args.input_dir, "*.bmp")))
    if not image_paths:
        print(f"No BMP images found in: {args.input_dir}")
        sys.exit(1)

    labeler = CircleLabeler(image_paths=image_paths, output_dir=args.output_dir)

    if not labeler.load_current_image():
        print("No images to load.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, labeler.mouse_callback)

    while True:
        cv2.imshow(WINDOW_NAME, labeler.display_image)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Exiting.")
            break

        elif key == 32:  # Space
            labeler.save_current_labels()
            has_next = labeler.go_to_next_image()
            if not has_next:
                print("All images processed.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
