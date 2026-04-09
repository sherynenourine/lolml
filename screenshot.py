import argparse
import time
from datetime import datetime
from PIL import ImageGrab


def save_screenshot(prefix="screenshot"):
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bmp"
    print("Taking screenshot now...")
    img = ImageGrab.grab()
    img.save(filename, "BMP")
    print(f"Screenshot saved as {filename}")


def main():
    parser = argparse.ArgumentParser(description="Take screenshots and save them as BMP files.")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds before taking the screenshot")
    parser.add_argument("--wait-enter", action="store_true", help="Wait for Enter before taking the screenshot")
    parser.add_argument("--interval", type=float, help="Take screenshots every N seconds")
    parser.add_argument("--count", type=int, default=1, help="Number of screenshots to take with --interval")
    parser.add_argument("--prefix", default="screenshot", help="Output filename prefix")
    args = parser.parse_args()

    print(f"Parsed args: {args}")

    if args.wait_enter:
        input("Press Enter to take the screenshot...")

    if args.delay > 0:
        print(f"Waiting {args.delay} seconds...")
        time.sleep(args.delay)

    if args.interval is not None:
        for i in range(args.count):
            print(f"Capture {i + 1}/{args.count}")
            save_screenshot(args.prefix)
            if i < args.count - 1:
                print(f"Sleeping {args.interval} seconds before next capture...")
                time.sleep(args.interval)
    else:
        save_screenshot(args.prefix)


if __name__ == "__main__":
    main()
