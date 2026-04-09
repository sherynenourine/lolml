from PIL import ImageGrab
from datetime import datetime

filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bmp"

img = ImageGrab.grab()
img.save(filename, "BMP")

print(f"Screenshot saved as {filename}")
