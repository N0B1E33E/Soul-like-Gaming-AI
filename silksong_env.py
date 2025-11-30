"""
Silksong 游戏环境 - 1280x720 版本
集成了 Boss HP 和 Player HP 检测器
"""

import time
import cv2
import numpy as np
import mss
from collections import deque
import pydirectinput

# 导入检测器
from boss_hp_detector import BossHPDetector
from player_hp_detector import PlayerHPDetector

# 动作空间（正确的按键）
# 跳跃：z
# 攻击：x  
# 冲刺：c
ACTIONS = {
    0: [],                    # NO_OP（什么都不做）
    1: [('left', 0.15)],      # 左移
    2: [('right', 0.15)],     # 右移
    3: [('z', 0.10)],         # 跳跃
    4: [('x', 0.10)],         # 攻击
    5: [('c', 0.08)],         # 冲刺
    6: [('left', 0.15), ('z', 0.10)],   # 左跳
    7: [('right', 0.15), ('z', 0.10)],  # 右跳
    8: [('x', 0.10), ('z', 0.10)],      # 跳攻
    9: [('left', 0.15), ('x', 0.10)],   # 左攻
}

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False


class SilksongEnv:
    """
    Silksong 游戏环境
    参考 seermer 的设计，适配 PPO 训练
    """
    
    def __init__(self, max_player_hp=6, boss_hp_max=800, frame_stack=4):
        # 截图区域（包含标题栏，height=800 确保底部不被截）
        self.monitor = {'left': 0, 'top': 0, 'width': 1280, 'height': 800}
        self.sct = mss.mss()
        
        # 检测器
        self.boss_detector = BossHPDetector()
        self.player_detector = PlayerHPDetector()
        
        # 参数
        self.max_player_hp = max_player_hp
        self.boss_hp_max = boss_hp_max
        self.frame_stack = frame_stack
        
        # 状态
        self.obs_stack = deque(maxlen=frame_stack)
        self.prev_player_hp = None
        self.prev_boss_hp = None
        self.timestep = 0
        
        # 奖励权重（激进版 - 快速学习）
        self.w_hurt = 0.5      # 受伤惩罚（降低，让 Agent 更勇敢）
        self.w_hit = 2.0       # 击中奖励（提高，大力鼓励攻击）
        self.w_idle = -8e-5    # 无事发生的小惩罚
        
        print("环境初始化完成!")
        print(f"动作空间: {len(ACTIONS)} 个动作")
        print(f"  0=NO_OP, 1=左, 2=右, 3=跳, 4=攻, 5=冲刺")
        print(f"  6=左跳, 7=右跳, 8=跳攻, 9=左攻")
        print(f"观察空间: ({frame_stack}, 128, 128)")
    
    def _capture_frame(self):
        """截图"""
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot)[:, :, :3]  # BGR
        return frame
    
    def _preprocess_frame(self, frame):
        """预处理为 128x128 灰度图（增加观察信息）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _execute_action(self, action):
        """执行动作"""
        macro = ACTIONS.get(action, [])
        for key, duration in macro:
            pydirectinput.keyDown(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            time.sleep(0.01)
    
    def reset(self):
        """重置环境"""
        print("\n重置环境...")
        
        # 等待游戏重新开始（你可能需要手动重启战斗）
        time.sleep(2.0)
        
        # 清空观察栈
        self.obs_stack.clear()
        
        # 初始观察
        frame = self._capture_frame()
        obs = self._preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.obs_stack.append(obs)
        
        # 初始状态
        self.prev_player_hp = self.max_player_hp
        self.prev_boss_hp = self.boss_hp_max
        self.timestep = 0
        
        return np.array(self.obs_stack, dtype=np.float32)
    
    def step(self, action):
        """
        执行一步
        
        参数：
            action: 0-7 的动作编号
        
        返回：
            obs: (4, 84, 84) 观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 执行动作
        self._execute_action(action)
        time.sleep(0.17)  # 参考 seermer 的 gap
        
        # 截图
        frame = self._capture_frame()
        obs = self._preprocess_frame(frame)
        self.obs_stack.append(obs)
        
        # 检测血量
        player_hp = self.player_detector.detect(frame)
        
        # Boss HP 检测（每 3 步一次，减少 OCR 开销但保持准确性）
        if self.timestep % 3 == 0 or self.timestep == 0:
            boss_hp_cur, boss_hp_max, boss_ok = self.boss_detector.detect(frame)
            
            # 处理检测失败
            if not boss_ok or boss_hp_cur is None:
                boss_hp_cur = self.prev_boss_hp
            else:
                self.boss_hp_max = boss_hp_max
                self.last_detected_boss_hp = boss_hp_cur  # 缓存
        else:
            # 使用缓存值
            boss_hp_cur = getattr(self, 'last_detected_boss_hp', self.prev_boss_hp)
            boss_ok = True
        
        # 计算伤害（参考 seermer）
        hurt = (player_hp < self.prev_player_hp)
        hit = (boss_hp_cur < self.prev_boss_hp)
        
        # 计算奖励（参考 seermer）
        reward = 0.0
        reward -= self.w_hurt * hurt     # 受伤：-0.8
        reward += self.w_hit * hit       # 击中：+0.5
        
        if not (hurt or hit):
            reward += self.w_idle  # 无事发生：-0.00008
        
        # 限制范围
        reward = np.clip(reward, -1.5, 1.5)
        
        # 判断结束（更严格的检查）
        win = False
        lose = False
        
        # 胜利：Boss HP 确实为 0 且检测成功
        if boss_ok and boss_hp_cur <= 0:
            win = True
        
        # 失败：Player HP 为 0（连续确认）
        if player_hp <= 0:
            # 额外检查：屏幕是否变暗（死亡画面）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            if brightness < 30:  # 确实很暗
                lose = True
            elif self.timestep > 5:  # 超过5步后相信检测
                lose = True
        
        done = win or lose
        
        # 额外奖励
        if win:
            reward += 10.0
        elif lose:
            reward -= 5.0
        
        # 更新状态
        if not done:
            self.prev_player_hp = player_hp
            self.prev_boss_hp = boss_hp_cur
        
        self.timestep += 1
        
        # 信息
        info = {
            'player_hp': player_hp,
            'boss_hp': boss_hp_cur,
            'hurt': hurt,
            'hit': hit,
            'timestep': self.timestep,
            'boss_detection_ok': boss_ok
        }
        
        # 打印
        if self.timestep % 10 == 0:
            print(f"t={self.timestep:3d} Player={player_hp}/{self.max_player_hp} "
                  f"Boss={boss_hp_cur:.0f}/{self.boss_hp_max:.0f} "
                  f"R={reward:.3f} OCR={'✓' if boss_ok else '✗'}")
        
        # Episode 结束时详细打印
        if done:
            reason = "WIN" if win else "LOSE"
            print(f"\n{'='*50}")
            print(f"Episode 结束 [{reason}] @ Step {self.timestep}")
            print(f"  Player HP: {player_hp}/{self.max_player_hp}")
            print(f"  Boss HP: {boss_hp_cur:.0f}/{self.boss_hp_max:.0f}")
            print(f"  Boss 伤害: {self.boss_hp_max - boss_hp_cur:.0f}")
            print(f"  Boss 检测: {'成功' if boss_ok else '失败'}")
            print(f"{'='*50}\n")
            info['result'] = 'win' if win else 'lose'
        
        return np.array(self.obs_stack, dtype=np.float32), reward, done, info
    
    def close(self):
        """关闭环境"""
        self.sct.close()


# 测试
if __name__ == '__main__':
    print("测试环境...")
    
    env = SilksongEnv()
    
    print("\n按 Enter 开始测试...")
    input()
    
    # 重置
    obs = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    # 运行几步
    for i in range(20):
        action = np.random.randint(0, 10)  # 0-9 共10个动作
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\nEpisode 结束: {info}")
            break
    
    env.close()
    print("\n测试完成!")