import cv2
import numpy as np
from mss import mss
import time

# Screenshot
print("Screenshot after 3 sec...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

with mss() as sct:
    monitor = {'left': 0, 'top': 0, 'width': 1280, 'height': 800}
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

print("Screenshot Done")
cv2.imwrite('screenshot.png', screenshot)

# Select region
print("\nSelect the Boss HP number and press SPACE to confirm")
x, y, w, h = cv2.selectROI("Boss HP", screenshot, True, False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No selection made, exit")
    exit()

x1, y1, x2, y2 = x, y, x+w, y+h
print(f"Region: ({x1},{y1}) -> ({x2},{y2})")

# Generate detector
code = f'''import cv2
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"D:\\Tesseract-OCR\\tesseract.exe"

class BossHPDetector:
    def __init__(self):
        self.x1 = {x1}
        self.y1 = {y1}
        self.x2 = {x2}
        self.y2 = {y2}
    
    def detect(self, screenshot):
        roi = screenshot[self.y1:self.y2, self.x1:self.x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, None, fx=4, fy=4)
        _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.replace('O', '0').replace('I', '1').replace(' ', '')
        match = re.search(r'(\\d+)/(\\d+)', text)
        if match:
            return int(match.group(1)), int(match.group(2)), True
        return None, None, False
'''

with open('boss_hp_detector.py', 'w') as f:
    f.write(code)

print("Generate: boss_hp_detector.py")
print("Done")