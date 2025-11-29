import cv2
import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

class BossHPDetector:
    def __init__(self):
        self.x1 = 598
        self.y1 = 709
        self.x2 = 682
        self.y2 = 725
    
    def detect(self, screenshot):
        roi = screenshot[self.y1:self.y2, self.x1:self.x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        big = cv2.resize(gray, None, fx=4, fy=4)
        _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789/')
        text = text.replace('O', '0').replace('I', '1').replace(' ', '')
        match = re.search(r'(\d+)/(\d+)', text)
        if match:
            return int(match.group(1)), int(match.group(2)), True
        return None, None, False
