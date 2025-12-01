# ocr_calibrator.py
# 校准 Boss 数字 OCR 的 ROI。支持两种模式：
#  1) ratio：相对 BOSS_HP_REGION 的比例带
#  2) peak ：以血条亮度“峰值行”为锚点，向上/向下像素偏移
# 运行：
#   python ocr_calibrator.py           # GUI 预览
#   python ocr_calibrator.py --nogui   # 无GUI，每 interval 秒落图
# 保存：按 S 写入 ocr_band.json
# 切换模式：按 M

import os
import json
import time
import argparse
from datetime import datetime

import cv2
import mss
import numpy as np

# ==== 尝试从 game_env 导入屏幕与血条区域 ====
try:
    from game_env import CAPTURE_REGION, BOSS_HP_REGION
except Exception as e:
    print("[Warn] 未能从 game_env 导入 CAPTURE_REGION/BOSS_HP_REGION：", e)
    # 兜底（1920x1080 全屏 & 常用血条 ROI）
    CAPTURE_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    BOSS_HP_REGION = {
        "top": int(0.88 * 1080),
        "left": int(0.263 * 1920),
        "width": int(0.477 * 1920),
        "height": int(0.023 * 1080),
    }

CFG_PATH = "ocr_band.json"
OUT_DIR = r"D:\python\NU\ocr_debug"

# ====== 默认参数（可改成你的习惯起点）======
DEFAULT_CFG = {
    "mode": "ratio",                  # "ratio" or "peak"
    "ratio": {
        "x_left": 0.35,               # 相对 BOSS_HP_REGION 的横向比例
        "x_right": 0.65,
        "y_top": 0.894,               # 你提出的更靠上/贴条参数
        "y_bottom": 1.03
    },
    "peak": {
        "x_center": 0.50,             # 峰值模式下，横向取中间带（中心+半宽）
        "half_width": 0.15,           # 即左右各 15% 条宽（等效 ~0.35~0.65）
        "top_offset_px": -2,          # 峰值行向上偏移（负数向上）
        "bottom_offset_px": 12        # 峰值行向下偏移（正数向下，含数字）
    }
}

def load_cfg():
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 补全缺省
        cfg = DEFAULT_CFG.copy()
        cfg["ratio"].update(data.get("ratio", {}))
        cfg["peak"].update(data.get("peak", {}))
        if data.get("mode") in ("ratio", "peak"):
            cfg["mode"] = data["mode"]
        return cfg
    return DEFAULT_CFG.copy()

def save_cfg(cfg):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("[Save] 写入参数到", CFG_PATH)

def grab_frame(sct):
    shot = sct.grab(CAPTURE_REGION)
    return np.array(shot)[:, :, :3]

def safe_rect(x0, y0, x1, y1, W, H):
    x0 = max(0, min(x0, W-1)); x1 = max(x0+1, min(x1, W))
    y0 = max(0, min(y0, H-1)); y1 = max(y0+1, min(y1, H))
    return x0, y0, x1, y1

def get_bar_roi(frame):
    bx, by, bw, bh = BOSS_HP_REGION["left"], BOSS_HP_REGION["top"], BOSS_HP_REGION["width"], BOSS_HP_REGION["height"]
    return frame[by:by+bh, bx:bx+bw], bx, by, bw, bh

def find_bar_peak_row(roi_bar):
    """在 BOSS_HP_REGION 内找亮度最大的行（血条中心线近似）"""
    if roi_bar.size == 0:
        return 0
    g = cv2.cvtColor(roi_bar, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_means = bw.mean(axis=1)
    return int(np.argmax(row_means))

def preprocess_bw(gray_roi):
    """为 OCR 做对比增强与二值化（便于观察 strip 是否干净）"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray_roi)
    g = cv2.GaussianBlur(g, (0,0), 1.0)
    g = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0,0), 2.0), -0.8, 0)
    up = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k1, 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2, 1)
    return bw

def calc_rect_ratio(cfg, bx, by, bw, bh, W, H):
    R = cfg["ratio"]
    x0 = int(bx + R["x_left"]  * bw)
    x1 = int(bx + R["x_right"] * bw)
    y0 = int(by + R["y_top"]   * bh)
    y1 = int(by + R["y_bottom"]* bh)
    return safe_rect(x0, y0, x1, y1, W, H)

def calc_rect_peak(cfg, roi_bar, bx, by, bw, bh, W, H):
    P = cfg["peak"]
    r_peak = find_bar_peak_row(roi_bar)
    x_center = P["x_center"]; halfw = P["half_width"]
    x0 = int(bx + (x_center - halfw) * bw)
    x1 = int(bx + (x_center + halfw) * bw)
    y0 = by + max(0, r_peak + P["top_offset_px"])
    y1 = by + min(bh-1, r_peak + P["bottom_offset_px"])
    return safe_rect(x0, y0, x1, y1, W, H)

def draw_text(img, text, x=20, y=40, color=(0,255,255), scale=1.1, thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nogui", action="store_true", help="无GUI模式：每 interval 秒保存预览")
    parser.add_argument("--interval", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = load_cfg()
    sct = mss.mss()
    W, H = CAPTURE_REGION["width"], CAPTURE_REGION["height"]

    # 步长
    step_small = 0.005   # ratio 模式下比例微调
    step_big   = 0.02
    px_small   = 1       # peak 模式下像素微调
    px_big     = 3

    print(
        "========== OCR Calibrator ==========\n"
        "模式切换：M （ratio <-> peak）\n"
        "\n[ratio 模式]（相对血条ROI比例）\n"
        "  H/J : x_left  → 右/左\n"
        "  K/L : x_right → 左/右\n"
        "  U/I : y_top   → 上/下\n"
        "  O/P : y_bottom→ 上/下（变薄/变厚）\n"
        "  大写(Shift) = 大步长\n"
        "\n[peak 模式]（相对峰值行像素偏移）\n"
        "  A/D : x_center 左/右   W/S : half_width 增/减\n"
        "  R/F : top_offset_px    上/下（负=向上）\n"
        "  T/G : bottom_offset_px 上/下（正=向下）\n"
        "  大写(Shift) = 大步长\n"
        "\n通用：S 保存到 ocr_band.json 并导出预览；R 恢复默认；Q 退出\n"
        "====================================\n"
    )

    use_gui = not args.nogui
    if use_gui:
        try:
            cv2.namedWindow("OCR Band Preview", cv2.WINDOW_NORMAL)
            cv2.namedWindow("OCR Strip (BW)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("OCR Band Preview", 1200, 700)
            cv2.resizeWindow("OCR Strip (BW)", 900, 260)
        except Exception as e:
            print("[Warn] GUI 初始化失败，切换 --nogui：", e)
            use_gui = False

    last_dump = 0.0
    while True:
        frame = grab_frame(sct)
        roi_bar, bx, by, bw, bh = get_bar_roi(frame)

        if cfg["mode"] == "ratio":
            x0, y0, x1, y1 = calc_rect_ratio(cfg, bx, by, bw, bh, W, H)
        else:
            x0, y0, x1, y1 = calc_rect_peak(cfg, roi_bar, bx, by, bw, bh, W, H)

        # 预处理 strip
        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bwimg = preprocess_bw(gray)

        # 叠加可视信息
        preview = frame.copy()
        # 血条框(白) + OCR 带(黄)
        cv2.rectangle(preview, (bx, by), (bx+bw, by+bh), (255,255,255), 1)
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0,255,255), 2)

        # HUD
        if cfg["mode"] == "ratio":
            tip1 = f"MODE=ratio   xL={cfg['ratio']['x_left']:.3f}  xR={cfg['ratio']['x_right']:.3f}"
            tip2 = f"yT={cfg['ratio']['y_top']:.3f}  yB={cfg['ratio']['y_bottom']:.3f}"
        else:
            tip1 = f"MODE=peak    xC={cfg['peak']['x_center']:.3f}  halfW={cfg['peak']['half_width']:.3f}"
            tip2 = f"topOff={cfg['peak']['top_offset_px']}px  botOff={cfg['peak']['bottom_offset_px']}px"
        draw_text(preview, tip1, 20, 40)
        draw_text(preview, tip2, 20, 80)

        if use_gui:
            cv2.imshow("OCR Band Preview", preview)
            cv2.imshow("OCR Strip (BW)", bwimg)
            key = cv2.waitKey(1) & 0xFF

            big_ratio = key in map(ord, "HJKLUIOP")
            big_peak  = key in map(ord, "ASDFTG")  # 大写为大步长
            # === ratio 模式 ===
            if cfg["mode"] == "ratio":
                step = step_big if big_ratio else step_small
                R = cfg["ratio"]
                if key in (ord('h'), ord('H')): R["x_left"]  = min(R["x_left"] + step, R["x_right"] - 0.02)
                elif key in (ord('j'), ord('J')): R["x_left"]  = max(R["x_left"] - step, 0.02)
                elif key in (ord('k'), ord('K')): R["x_right"] = max(R["x_right"] - step, R["x_left"] + 0.02)
                elif key in (ord('l'), ord('L')): R["x_right"] = min(R["x_right"] + step, 0.98)
                elif key in (ord('u'), ord('U')): R["y_top"]    = max(R["y_top"] - step, 0.70)
                elif key in (ord('i'), ord('I')): R["y_top"]    = min(R["y_top"] + step, R["y_bottom"] - 0.01)
                elif key in (ord('o'), ord('O')): R["y_bottom"] = max(R["y_bottom"] - step, R["y_top"] + 0.01)
                elif key in (ord('p'), ord('P')): R["y_bottom"] = min(R["y_bottom"] + step, 2.20)

            # === peak 模式 ===
            else:
                step = px_big if big_peak else px_small
                P = cfg["peak"]
                if key in (ord('a'), ord('A')): P["x_center"]   = max(0.05, P["x_center"] - 0.01)
                elif key in (ord('d'), ord('D')): P["x_center"] = min(0.95, P["x_center"] + 0.01)
                elif key in (ord('w'), ord('W')): P["half_width"] = min(0.45, P["half_width"] + 0.01)
                elif key in (ord('s'), ord('S')): P["half_width"] = max(0.02, P["half_width"] - 0.01)
                elif key in (ord('r'), ord('R')): P["top_offset_px"]    -= step
                elif key in (ord('f'), ord('F')): P["top_offset_px"]    += step
                elif key in (ord('t'), ord('T')): P["bottom_offset_px"] -= step
                elif key in (ord('g'), ord('G')): P["bottom_offset_px"] += step

            # 通用：保存、模式切换、恢复默认、退出
            if key in (ord('m'), ord('M')):
                cfg["mode"] = "peak" if cfg["mode"] == "ratio" else "ratio"
                print("[Mode] ->", cfg["mode"])
            elif key in (ord('s'), ord('S')):   # 在 peak 模式下小写 s 会被上面用掉，所以也支持这里
                save_cfg(cfg)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(OUT_DIR, f"box_preview_{stamp}.jpg"), preview)
                cv2.imwrite(os.path.join(OUT_DIR, f"strip_bw_{stamp}.png"), bwimg)
                print("[Save] 预览导出到", OUT_DIR)
            elif key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('z'), ord('Z')):   # 恢复默认（避免和 peak 的 R 冲突）
                cfg = DEFAULT_CFG.copy()
                print("[Reset] 恢复默认:", cfg)

        else:
            now = time.time()
            if now - last_dump > args.interval:
                last_dump = now
                stamp = datetime.now().strftime("%H%M%S")
                cv2.imwrite(os.path.join(OUT_DIR, f"nogui_preview_{stamp}.jpg"), preview)
                cv2.imwrite(os.path.join(OUT_DIR, f"nogui_strip_{stamp}.png"), bwimg)
                save_cfg(cfg)
                print("[NoGUI] 保存一帧预览，当前配置：", cfg)

    if use_gui:
        cv2.destroyAllWindows()
    save_cfg(cfg)
    print("[Exit] 已保存最终配置到", CFG_PATH)

if __name__ == "__main__":
    main()
