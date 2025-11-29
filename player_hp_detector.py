import numpy as np

class PlayerHPDetector:
    def __init__(self):
        self.y = 108
        self.x_positions = np.array([139, 171, 204, 238, 270, 303])
        self.max_hp = 6
    
    def detect(self, screenshot):
        line = screenshot[self.y, :, 0]
        pixels = line[self.x_positions]
        hp = (pixels > 150).sum()
        return int(hp)
    
    def detect_ratio(self, screenshot):
        return self.detect(screenshot) / self.max_hp
