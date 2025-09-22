import time
import cv2
import numpy as np
import mss
from game_env import CAPTURE_REGION, PLAYER_HP_REGION, BOSS_HP_REGION

def build_views(img):
    canvas = img.copy()

    def draw_rect(r, color):
        x,y,w,h = r["left"], r["top"], r["width"], r["height"]
        cv2.rectangle(canvas, (x,y), (x+w,y+h), color, 2)

    draw_rect(PLAYER_HP_REGION, (255,0,0))  # blue for player
    draw_rect(BOSS_HP_REGION,   (0,0,255))  # red for boss

    def heat_of_player(img, r):
        x,y,w,h = r["left"], r["top"], r["width"], r["height"]
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV,21,5)
        vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        out = img.copy(); out[y:y+h,x:x+w]=vis; return out

    def heat_of_boss(img, r):
        x,y,w,h = r["left"], r["top"], r["width"], r["height"]
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        _, bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        out = img.copy(); out[y:y+h,x:x+w]=vis; return out

    pl_view = heat_of_player(img, PLAYER_HP_REGION)
    bo_view = heat_of_boss(img, BOSS_HP_REGION)

    row1 = cv2.resize(canvas, (1280, 720))
    row2 = cv2.resize(pl_view, (1280, 720))
    row3 = cv2.resize(bo_view, (1280, 720))
    mosaic = np.vstack([row1, row2, row3])
    return mosaic

def main(save_path="preview_mosaic.jpg", interval=0.5):
    sct = mss.mss()
    print(f"No GUI mode: preview saved to {save_path} every {interval}s. Ctrl+C to stop.")
    while True:
        shot = sct.grab(CAPTURE_REGION)
        img = np.array(shot)[:, :, :3]
        mosaic = build_views(img)
        cv2.imwrite(save_path, mosaic)
        print(time.strftime("[%H:%M:%S] saved -> "), save_path)
        time.sleep(interval)

if __name__ == "__main__":
    main()
