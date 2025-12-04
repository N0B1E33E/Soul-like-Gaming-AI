import cv2
import numpy as np
from mss import mss
import time

# Capture screenshot after countdown
print("3 seconds until screenshot...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

with mss() as sct:
    monitor = {'left': 0, 'top': 0, 'width': 1280, 'height': 800}
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

print("Screenshot captured")
cv2.imwrite('screenshot.png', screenshot)

# Select the player HP area
print("\nSelect the player HP area (top-left), press SPACE to confirm")
x, y, w, h = cv2.selectROI("Player HP Area", screenshot, True, False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No selection made, exiting")
    exit()

hp_region = screenshot[y:y+h, x:x+w]

# Display the area for clicking
print("\nNow click the center of each mask in the HP bar.")
print("Press any key when done")

clicks = []
display = hp_region.copy()

def mouse_callback(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append(mx + x)
        cv2.circle(display, (mx, my), 3, (0, 255, 0), -1)
        cv2.putText(display, str(len(clicks)), (mx-5, my-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Click masks", display)
        print(f"Mask {len(clicks)}: X={mx+x}")

cv2.namedWindow("Click masks")
cv2.setMouseCallback("Click masks", mouse_callback)
cv2.imshow("Click masks", display)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicks) == 0:
    print("No clicks detected, exiting")
    exit()

y_coord = y + h // 2
x_coords = sorted(clicks)

print(f"\nY={y_coord}, X={x_coords}")

# 生成检测器
code = f'''import numpy as np

class PlayerHPDetector:
    def __init__(self):
        self.y = {y_coord}
        self.x_positions = np.array({x_coords})
        self.max_hp = {len(x_coords)}
    
    def detect(self, screenshot):
        line = screenshot[self.y, :, 0]
        pixels = line[self.x_positions]
        hp = (pixels > 200).sum()
        return int(hp)
    
    def detect_ratio(self, screenshot):
        return self.detect(screenshot) / self.max_hp
'''

with open('player_hp_detector.py', 'w') as f:
    f.write(code)

print("Generated: player_hp_detector.py")
print("Done!")
