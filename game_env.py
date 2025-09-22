# -*- coding: utf-8 -*-
import time
import os
import re
import json
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Union

import cv2
import mss
import numpy as np
import pydirectinput
import pytesseract

# —— 若 Tesseract 未在 PATH，请取消下一行注释并改为你的安装路径 ——
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== Screen & Regions (2560x1440 fullscreen) ==================
CAPTURE_REGION = {"top": 0, "left": 0, "width": 2560, "height": 1440}

# 玩家血量 ROI（你已调好）
_PLAYER_PIPS_FRACTION = dict(top=0.062, left=0.130, width=0.227, height=0.077)
# Boss 底部血条
_BOSS_BAR_FRACTION   = dict(top=0.89,  left=0.263, width=0.477, height=0.030)

def _fraction_to_rect(frac: dict) -> dict:
    return {
        "top":    int(frac["top"]    * CAPTURE_REGION["height"]),
        "left":   int(frac["left"]   * CAPTURE_REGION["width"]),
        "width":  int(frac["width"]  * CAPTURE_REGION["width"]),
        "height": int(frac["height"] * CAPTURE_REGION["height"]),
    }

PLAYER_HP_REGION = _fraction_to_rect(_PLAYER_PIPS_FRACTION)
BOSS_HP_REGION   = _fraction_to_rect(_BOSS_BAR_FRACTION)

# ===================== 读取 OCR 配置（ocr_band.json） =====================
_OCR_DEFAULT = {
    "mode": "peak",           # "ratio" or "peak"
    "ratio": {                # 仅当 mode="ratio" 有效（相对 BOSS_HP_REGION 的比例框）
        "x_left": 0.35, "x_right": 0.65,
        "y_top": 0.894, "y_bottom": 1.03
    },
    "peak": {                 # 我们推荐使用的模式（条中央亮度峰行为锚点）
        "x_center": 0.50, "half_width": 0.07,
        "top_offset_px": -1, "bottom_offset_px": 24
    }
}
_OCR_CFG_PATH = "ocr_band.json"

def load_ocr_cfg(path: str = _OCR_CFG_PATH) -> dict:
    cfg = json.loads(json.dumps(_OCR_DEFAULT))  # deep copy
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "mode" in data and data["mode"] in ("ratio", "peak"):
                    cfg["mode"] = data["mode"]
                if "ratio" in data and isinstance(data["ratio"], dict):
                    cfg["ratio"].update(data["ratio"])
                if "peak" in data and isinstance(data["peak"], dict):
                    cfg["peak"].update(data["peak"])
            print("[OCR] loaded ocr_band.json:", cfg)
        except Exception as e:
            print("[OCR] load failed, use default:", e)
    else:
        print("[OCR] ocr_band.json not found, use default:", cfg)
    return cfg

# ===================== Action macros (支持组合键) =====================
ACTION_SPACE: Dict[int, List[Tuple[Union[str, List[str], Tuple[str, ...]], float]]] = {
    0: [],                                  # noop
    1: [("left", 0.10)],
    2: [("left", 0.50)],
    3: [("left", 1.00)],
    4: [("right", 0.10)],
    5: [("right", 0.50)],
    6: [("right", 1.00)],
    7: [("z", 0.50)],                       # jump_long
    8: [("c", 0.06)],                       # dash / dodge
    9: [("x", 0.04)],                       # light attack
    10:[("z", 0.20), ("x", 0.04)],          # jump-attack
    11:[("a", 0.50)],                       # heal
    12:[("z", 0.50), ("z", 0.50)],          # double jump
    13:[("z", 0.20)],                       # jump_short
    14:[("z", 0.35)],                       # jump_mid
    15:[(["right","x"], 0.06)],             # right+attack
    16:[(["left","x"],  0.06), ("x", 0.02)],# left+attack
    17:[(["left","z"],  0.50)],             # left+jump
    18:[(["right","z"], 0.50)],             # right+jump
}

# ================== Input system tweaks ==================
pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

def _sleep_chunked(total: float, dt: float = 0.02):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < total:
        time.sleep(min(dt, total - (time.perf_counter() - t0)))

def press_macro(macro: List[Tuple[Union[str, List[str], Tuple[str, ...]], float]]):
    for key, dur in macro:
        if isinstance(key, (list, tuple)):           # 并行按键
            for k in key: pydirectinput.keyDown(k)
            _sleep_chunked(max(0.0, dur))
            for k in key: pydirectinput.keyUp(k)
            continue
        if dur >= 0.07:                               # 长按
            pydirectinput.keyDown(key)
            _sleep_chunked(dur)
            pydirectinput.keyUp(key)
        else:                                         # 轻点
            pydirectinput.press(key)
            if dur > 0: time.sleep(dur)

# ====================== Preprocess frames ======================
def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)

# ====================== HP estimators ======================
def estimate_player_pips_ratio(frame_bgr: np.ndarray, region: dict) -> float:
    x, y, w, h = region["left"], region["top"], region["width"], region["height"]
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    proj = th.sum(axis=1)
    row  = int(np.argmax(proj))
    band = th[max(0,row-2):min(h,row+3), :]
    xs = band.sum(axis=0)
    lit_len = int((xs > 0).sum())
    ratio = lit_len / float(w + 1e-6)
    return float(np.clip(ratio * 1.6, 0.0, 1.0))

def estimate_bossbar_ratio(frame_bgr: np.ndarray, region: dict) -> float:
    x, y, w, h = region["left"], region["top"], region["width"], region["height"]
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_means = bw.mean(axis=1)
    r = int(np.argmax(row_means))
    row = (bw[r, :] > 127).astype(np.uint8)
    run = 0
    for v in row:
        if v == 1: run += 1
        else: break
    ratio = run / float(w + 1e-6)
    return float(np.clip(ratio, 0.0, 1.0))

def _safe_rect_xyxy(x0, y0, x1, y1, W, H):
    x0 = max(0, min(x0, W-1)); x1 = max(x0+1, min(x1, W))
    y0 = max(0, min(y0, H-1)); y1 = max(y0+1, min(y1, H))
    return x0, y0, x1, y1

# ====================== OCR 辅助 ======================
_OCR_CFGS = [
    r'--oem 3 --psm 7  --dpi 300 -c tessedit_char_whitelist=0123456789/',
    r'--oem 1 --psm 13 --dpi 300 -c tessedit_char_whitelist=0123456789/'
]

def _postfix_clean(txt: str) -> str:
    tbl = str.maketrans({'O':'0','o':'0','Q':'0','I':'1','l':'1','|':'1','!':'1','S':'5','B':'8'})
    return txt.translate(tbl)

def _find_bar_peak_row(frame_bgr: np.ndarray, bar_region: dict) -> int:
    bx, by, bw, bh = bar_region["left"], bar_region["top"], bar_region["width"], bar_region["height"]
    roi = frame_bgr[by:by+bh, bx:bx+bw]
    if roi.size == 0:
        return 0
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    _, bwimg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_means = bwimg.mean(axis=1)
    return int(np.argmax(row_means))  # 亮度峰值 ≈ 条中心线

def _build_ocr_roi_xyxy(frame_bgr: np.ndarray, bar_region: dict, ocr_cfg: dict):
    H, W = frame_bgr.shape[:2]
    bx, by, bw, bh = bar_region["left"], bar_region["top"], bar_region["width"], bar_region["height"]

    mode = ocr_cfg.get("mode", "peak")
    if mode == "ratio":
        R = ocr_cfg["ratio"]
        x0 = int(bx + R["x_left"]  * bw)
        x1 = int(bx + R["x_right"] * bw)
        y0 = int(by + R["y_top"]   * bh)
        y1 = int(by + R["y_bottom"]* bh)
        return _safe_rect_xyxy(x0, y0, x1, y1, W, H)

    # peak 模式
    P = ocr_cfg["peak"]
    r_peak = _find_bar_peak_row(frame_bgr, bar_region)
    x_center, halfw = P["x_center"], P["half_width"]
    x0 = int(bx + (x_center - halfw) * bw)
    x1 = int(bx + (x_center + halfw) * bw)
    y0 = by + max( 0, r_peak + int(P["top_offset_px"]) )
    y1 = by + min( bh-1, r_peak + int(P["bottom_offset_px"]) )
    return _safe_rect_xyxy(x0, y0, x1, y1, W, H)

def read_boss_hp_ocr(frame_bgr: np.ndarray, bar_region: dict, ocr_cfg: dict,
                     expect_max: int = 1800, debug_dir: str | None = "ocr_debug"):
    """
    返回 (hp_abs, hp_max, visible, raw_text)；失败 -> (None, expect_max, False, raw_text)
    """
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _build_ocr_roi_xyxy(frame_bgr, bar_region, ocr_cfg)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None, float(expect_max), False, ""

    # 预处理
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (0,0), 1.0)
    g = cv2.addWeighted(g, 1.8, cv2.GaussianBlur(g, (0,0), 2.0), -0.8, 0)
    up = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k1, 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2, 1)

    raw = ""
    for cfg in _OCR_CFGS:
        raw = pytesseract.image_to_string(bw, config=cfg).strip()
        txt = _postfix_clean(raw)
        m = re.search(r'(\d{2,4})\s*[/\\]\s*(\d{3,4})', txt)
        if m:
            cur, mx = int(m.group(1)), int(m.group(2))
            if 0 <= cur <= mx and 200 <= mx <= 10000:
                if debug_dir:
                    try:
                        os.makedirs(debug_dir, exist_ok=True)
                        dbg = frame_bgr.copy()
                        cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,255), 2)
                        cv2.imwrite(os.path.join(debug_dir, "box_preview.png"), dbg)
                    except Exception:
                        pass
                return float(cur), float(mx), True, raw

    # 失败：落图
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(debug_dir, f"roi_{stamp}.png"), roi)
            cv2.imwrite(os.path.join(debug_dir, f"bin_{stamp}.png"), bw)
            dbg = frame_bgr.copy()
            cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,255), 2)
            cv2.imwrite(os.path.join(debug_dir, "box_preview.png"), dbg)
        except Exception:
            pass

    return None, float(expect_max), False, raw

# ====================== FPS Sync ======================
class FPSSync:
    def __init__(self, target_fps=15):
        self.dt = 1.0 / target_fps
        self.t0 = time.perf_counter()
    def step(self):
        t = time.perf_counter()
        wait = self.dt - (t - self.t0)
        if wait > 0: time.sleep(wait)
        self.t0 = time.perf_counter()

# ====================== Environment ======================
class GameEnv:
    def __init__(self, stack_size: int = 4, frame_skip: int = 4, target_fps: int = 15,
                 use_auto_restart: bool = False, min_reset_interval: float = 5.0):
        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self.sync = FPSSync(target_fps)
        self.sct = mss.mss()
        self.obs_stack = deque(maxlen=stack_size)

        # 载入 OCR 配置
        self.ocr_cfg = load_ocr_cfg()

        # HP / 终止参数
        self.boss_hp_max = 1800.0
        self.min_steps_before_end_boss   = 60
        self.min_steps_before_end_player = 0
        self.need_k_frames_boss   = 12
        self.need_k_frames_player = 6

        # 可见性稳定计数（用 OCR 窄带判定，≥2 帧更灵敏）
        self._boss_visible_true_cnt = 0
        self._boss_visible_need_k   = 2

        # 实时打印
        self.print_boss_hp       = True
        self.print_every_steps   = 5
        self.print_min_delta_abs = 10.0
        self._last_print_hp_abs  = None
        self._last_print_step    = 0

        # 惩罚：重复动作 & 15s 未命中
        self.repeat_action_threshold = 5
        self.repeat_action_penalty   = 0.02
        self._last_action            = None
        self._repeat_action_count    = 0

        self.nohit_seconds         = 15.0
        self.nohit_penalty         = 1.0
        self._steps_since_hit      = 0
        self._nohit_step_threshold = int(self.nohit_seconds * (1.0 / self.sync.dt))

        # 其它
        self.prev_boss_hp   = None   # 绝对血量
        self.prev_player_hp = None   # 比例
        self.timestep = 0
        self.max_steps = 300 * target_fps

        self.use_auto_restart = use_auto_restart
        self._last_reset_ts = 0.0
        self.min_reset_interval = min_reset_interval

    # ---------------- API ----------------
    def reset(self) -> np.ndarray:
        # 每次 reset 时也尝试重载一次配置（方便边训边调）
        self.ocr_cfg = load_ocr_cfg()

        if self.use_auto_restart:
            now = time.perf_counter()
            if now - self._last_reset_ts >= self.min_reset_interval:
                print("[safe_restart] waiting 14s...")
                time.sleep(14.0)
                self._last_reset_ts = now

        self.obs_stack.clear()
        frame = self._grab_frame()
        obs = preprocess_frame(frame)
        for _ in range(self.stack_size):
            self.obs_stack.append(obs)

        self.prev_boss_hp   = float(self.boss_hp_max)  # 满血
        self.prev_player_hp = 1.0

        self._boss_visible_true_cnt = 0
        self.timestep       = 0

        self._last_print_hp_abs  = self.boss_hp_max
        self._last_print_step    = 0
        self._last_action         = None
        self._repeat_action_count = 0
        self._steps_since_hit     = 0

        return self._get_obs()

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}

        boss_hp_abs   = self.prev_boss_hp   if self.prev_boss_hp   is not None else float(self.boss_hp_max)
        player_hp_rat = self.prev_player_hp if self.prev_player_hp is not None else 1.0

        damage_this_step = 0.0
        taken_this_step  = 0.0

        # 连续同动作计数
        if self._last_action is None or self._last_action != action:
            self._last_action = action
            self._repeat_action_count = 1
        else:
            self._repeat_action_count += 1

        for _ in range(self.frame_skip):
            press_macro(ACTION_SPACE.get(action, []))
            self.sync.step()

            frame = self._grab_frame()
            obs = preprocess_frame(frame)
            self.obs_stack.append(obs)

            # ---------- ① 像素法估比例（兜底用，不用于可见性） ----------
            cur_boss_ratio = estimate_bossbar_ratio(frame, BOSS_HP_REGION)

            # ---------- ② 用 OCR 窄带判可见 ----------
            x0, y0, x1, y1 = _build_ocr_roi_xyxy(frame, BOSS_HP_REGION, self.ocr_cfg)
            roi_ocr = frame[y0:y1, x0:x1]
            boss_bar_visible = False
            if roi_ocr.size > 0:
                g_small = cv2.cvtColor(roi_ocr, cv2.COLOR_BGR2GRAY)
                up = cv2.resize(g_small, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                _, bin_inv = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                fg_frac = (bin_inv > 127).mean()  # 字形前景占比
                boss_bar_visible = (fg_frac > 0.04)  # ★ 阈值：可调 0.03~0.06

            # 稳定计数
            if boss_bar_visible:
                self._boss_visible_true_cnt = min(self._boss_visible_true_cnt + 1, 10)
            else:
                self._boss_visible_true_cnt = 0

            # ---------- ③ 仅在稳定可见时 OCR ----------
            hp_abs_ocr = None
            ocr_ok = False
            raw_txt = ""
            if self._boss_visible_true_cnt >= self._boss_visible_need_k:
                hp_abs_ocr, hp_max_ocr, ocr_ok, raw_txt = read_boss_hp_ocr(
                    frame, BOSS_HP_REGION, self.ocr_cfg,
                    expect_max=int(self.boss_hp_max),
                    debug_dir="ocr_debug"   # 调稳后可改 None
                )
                if ocr_ok:
                    self.boss_hp_max = hp_max_ocr
                    cur_boss_abs   = hp_abs_ocr
                    cur_boss_ratio = cur_boss_abs / max(1.0, self.boss_hp_max)

            # 不可见则强制视作满血，避免误判
            if not boss_bar_visible:
                cur_boss_ratio = 1.0

            # 转为绝对血量
            cur_boss_abs = cur_boss_ratio * self.boss_hp_max

            # —— 实时打印（节流）——
            if self.print_boss_hp:
                need_by_step  = (self.timestep - self._last_print_step) >= self.print_every_steps
                need_by_delta = (self._last_print_hp_abs is None) or (abs(cur_boss_abs - self._last_print_hp_abs) >= self.print_min_delta_abs)
                if need_by_step or need_by_delta:
                    print(f"[Boss HP] t={self.timestep:3d} visible={boss_bar_visible} text='{raw_txt}' "
                          f"ratio={cur_boss_ratio:.3f} abs={cur_boss_abs:.0f}/{self.boss_hp_max:.0f}")
                    self._last_print_hp_abs = cur_boss_abs
                    self._last_print_step   = self.timestep

            # ---------- Player 血量 ----------
            cur_player_ratio = estimate_player_pips_ratio(frame, PLAYER_HP_REGION)

            # 变化量
            dmg_abs = max(0.0, boss_hp_abs   - cur_boss_abs)
            tkn_rat = max(0.0, player_hp_rat - cur_player_ratio)

            damage_this_step += dmg_abs
            taken_this_step  += tkn_rat

            # 奖励
            r_damage  = 6.0 * dmg_abs
            r_survive = 0.008
            r_taken   = 7.0 * tkn_rat
            total_reward += float(r_damage + r_survive - r_taken)

            # 更新状态
            boss_hp_abs, player_hp_rat = cur_boss_abs, cur_player_ratio
            self.prev_boss_hp, self.prev_player_hp = boss_hp_abs, player_hp_rat
            self.timestep += 1

            # -------- 终止判定（玩家优先） --------
            allow_boss_end   = (self.timestep >= self.min_steps_before_end_boss)
            allow_player_end = (self.timestep >= self.min_steps_before_end_player)

            if boss_bar_visible and cur_boss_ratio <= 0.02:
                self.low_boss_cnt = min(self.low_boss_cnt + 1, 999)
            else:
                self.low_boss_cnt = 0

            if cur_player_ratio <= 0.05:
                self.low_player_cnt = min(self.low_player_cnt + 1, 999)
            else:
                self.low_player_cnt = 0

            if allow_player_end and self.low_player_cnt >= self.need_k_frames_player:
                total_reward -= 6.0
                done = True
                info["result"] = "player_down"
            elif allow_boss_end and self.low_boss_cnt >= self.need_k_frames_boss:
                total_reward += 6.0
                done = True
                info["result"] = "boss_down"
            elif self.timestep >= self.max_steps:
                done = True
                info["result"] = "timeout"

            if done:
                break

        # —— 步后惩罚 ——
        if self._repeat_action_count > self.repeat_action_threshold:
            over = self._repeat_action_count - self.repeat_action_threshold
            total_reward -= self.repeat_action_penalty * over

        if damage_this_step > 0.0:
            self._steps_since_hit = 0
        else:
            self._steps_since_hit += 1
        if self._steps_since_hit >= self._nohit_step_threshold:
            total_reward -= self.nohit_penalty
            self._steps_since_hit = 0

        # 回传 info
        info["damage"] = float(damage_this_step)
        info["taken"]  = float(taken_this_step)
        info["boss_ratio"]   = float(cur_boss_ratio)
        info["boss_hp_abs"]  = float(cur_boss_abs)
        info["boss_visible"] = bool(self._boss_visible_true_cnt > 0)

        if done:
            print(f"[Episode End] reason={info.get('result')} "
                  f"boss_visible={info['boss_visible']} "
                  f"boss_ratio={info['boss_ratio']:.3f} "
                  f"boss_hp_abs={info['boss_hp_abs']:.0f}/{self.boss_hp_max:.0f} "
                  f"player_hp={player_hp_rat:.3f} "
                  f"low_cnt(P/B)={self.low_player_cnt}/{self.low_boss_cnt} "
                  f"steps={self.timestep}")

        return self._get_obs(), total_reward, done, info

    # ---------------- Helpers ----------------
    def _grab_frame(self) -> np.ndarray:
        shot = self.sct.grab(CAPTURE_REGION)
        img = np.array(shot)[:, :, :3]
        return img

    def _get_obs(self) -> np.ndarray:
        return np.array(self.obs_stack, dtype=np.float32)
