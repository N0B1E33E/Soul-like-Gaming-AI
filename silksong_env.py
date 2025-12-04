"""
Silksong Game Environment - 1280x720 Resolution
Integrates Boss HP and Player HP detectors
"""

import time
import cv2
import numpy as np
import mss
from collections import deque
import pydirectinput

# Import detector
from boss_hp_detector import BossHPDetector
from player_hp_detector import PlayerHPDetector

# Action Dictionary
ACTIONS = {
    0: [],                    # NO_OP
    1: [('left', 0.15)],      # Move left
    2: [('right', 0.15)],     # Move right
    3: [('z', 0.10)],         # Jump
    4: [('x', 0.10)],         # Attack
    5: [('c', 0.08)],         # Dash
    6: [('left', 0.15), ('z', 0.10)],   # Left Jump
    7: [('right', 0.15), ('z', 0.10)],  # Right Jump
    8: [('x', 0.10), ('z', 0.10)],      # Jump Attack
    9: [('c', 0.15), ('x', 0.10)],   # Dash Attack
}

pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False


class SilksongEnv:
    """
    Silksong Game Environment
    """
    
    def __init__(self, max_player_hp=6, boss_hp_max=800, frame_stack=4):
        # Screenshot region
        self.monitor = {'left': 0, 'top': 0, 'width': 1280, 'height': 800}
        self.sct = mss.mss()
        
        # Detector
        self.boss_detector = BossHPDetector()
        self.player_detector = PlayerHPDetector()
        
        # Parameters
        self.max_player_hp = max_player_hp
        self.boss_hp_max = boss_hp_max
        self.frame_stack = frame_stack
        
        # State
        self.obs_stack = deque(maxlen=frame_stack)
        self.prev_player_hp = None
        self.prev_boss_hp = None
        self.timestep = 0
        
        # Reward Weight
        self.w_hurt = 0.5      # Hurt punishment（make Agent more brave）
        self.w_hit = 2.0       # Hit reward (greatly encouraging attacks)
        self.w_idle = -8e-5    # Small penalty for nothing happening
        
        print("Environment initialization complete!")
        print(f"Actions: {len(ACTIONS)} actions")
        print(f"  0=NO_OP, 1=Left, 2=Right, 3=Jump, 4=Attack, 5=Dash")
        print(f"  6=Left Jump, 7=Right Jump, 8=Jump Attack, 9=Dash Attack")
        print(f"Observation space: ({frame_stack}, 128, 128)")
    
    def _capture_frame(self):
        """Capture"""
        shot = self.sct.grab(self.monitor)
        frame = np.array(shot)[:, :, :3]  # BGR
        return frame
    
    def _preprocess_frame(self, frame):
        """Preprocessed into a 128x128 grayscale image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _execute_action(self, action):
        """Execute action"""
        macro = ACTIONS.get(action, [])
        for key, duration in macro:
            pydirectinput.keyDown(key)
            time.sleep(duration)
            pydirectinput.keyUp(key)
            time.sleep(0.01)
    
    def reset(self):
        """Reset Environment"""
        print("\nReset Environment...")
        
        # Waiting for the game to restart
        time.sleep(2.0)
        
        # Clear the observation stack
        self.obs_stack.clear()
        
        # Initial observation
        frame = self._capture_frame()
        obs = self._preprocess_frame(frame)
        for _ in range(self.frame_stack):
            self.obs_stack.append(obs)
        
        # Initial state
        self.prev_player_hp = self.max_player_hp
        self.prev_boss_hp = self.boss_hp_max
        self.timestep = 0
        
        return np.array(self.obs_stack, dtype=np.float32)
    
    def step(self, action):
        """
        Perform one action
        
        Parametes:
            action: number from 0-9
        
        返回：
            obs: (4, 84, 84)
            reward: Reward
            done: whether the process has ended
            info: Additional information
            
        """
        # Execute action
        self._execute_action(action)
        time.sleep(0.17)
        
        # Capture frame
        frame = self._capture_frame()
        obs = self._preprocess_frame(frame)
        self.obs_stack.append(obs)
        
        # Player health point detect
        player_hp = self.player_detector.detect(frame)
        
        # Boss health point detect
        if self.timestep % 3 == 0 or self.timestep == 0:
            boss_hp_cur, boss_hp_max, boss_ok = self.boss_detector.detect(frame)
            
            # Detection failure
            if not boss_ok or boss_hp_cur is None:
                boss_hp_cur = self.prev_boss_hp
            else:
                self.boss_hp_max = boss_hp_max
                self.last_detected_boss_hp = boss_hp_cur  # Cache
        else:
            # Use cached values
            boss_hp_cur = getattr(self, 'last_detected_boss_hp', self.prev_boss_hp)
            boss_ok = True
        
        # Calculate damage
        hurt = (player_hp < self.prev_player_hp)
        hit = (boss_hp_cur < self.prev_boss_hp)
        
        # Calculate rewards
        reward = 0.0
        reward -= self.w_hurt * hurt
        reward += self.w_hit * hit
        
        if not (hurt or hit):
            reward += self.w_idle
        
        # Limitations
        reward = np.clip(reward, -1.5, 1.5)
        
        # Check win or lose
        win = False
        lose = False
        
        # Win: Boss HP is indeed 0 and the detection was successful
        if boss_ok and boss_hp_cur <= 0:
            win = True
        
        # Failure: Player HP is 0
        if player_hp <= 0:
            # dditional check: Does the screen dim (death screen)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            if brightness < 30:
                lose = True
            elif self.timestep > 5:
                lose = True
        
        done = win or lose
        
        # Additional rewards
        if win:
            reward += 10.0
        elif lose:
            reward -= 5.0
        
        # Update status
        if not done:
            self.prev_player_hp = player_hp
            self.prev_boss_hp = boss_hp_cur
        
        self.timestep += 1
        
        # All infos
        info = {
            'player_hp': player_hp,
            'boss_hp': boss_hp_cur,
            'hurt': hurt,
            'hit': hit,
            'timestep': self.timestep,
            'boss_detection_ok': boss_ok
        }
        
        # Print
        if self.timestep % 10 == 0:
            print(f"t={self.timestep:3d} Player={player_hp}/{self.max_player_hp} "
                  f"Boss={boss_hp_cur:.0f}/{self.boss_hp_max:.0f} "
                  f"R={reward:.3f} OCR={'✓' if boss_ok else '✗'}")
        
        # Episode
        if done:
            reason = "WIN" if win else "LOSE"
            print(f"\n{'='*50}")
            print(f"Episode ended [{reason}] @ Step {self.timestep}")
            print(f"  Player HP: {player_hp}/{self.max_player_hp}")
            print(f"  Boss HP: {boss_hp_cur:.0f}/{self.boss_hp_max:.0f}")
            print(f"  Boss damage: {self.boss_hp_max - boss_hp_cur:.0f}")
            print(f"  Boss Detection: {'Success' if boss_ok else 'Failure'}")
            print(f"{'='*50}\n")
            info['result'] = 'win' if win else 'lose'
        
        return np.array(self.obs_stack, dtype=np.float32), reward, done, info
    
    def close(self):
        """Close environment"""
        self.sct.close()


# Test
if __name__ == '__main__':
    print("Test environment...")
    
    env = SilksongEnv()
    
    print("\nPress Enter to start the test...")
    input()
    
    # Reset
    obs = env.reset()
    print(f"Obs shape: {obs.shape}")
    
    # Run a few steps
    for i in range(20):
        action = np.random.randint(0, 10)
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\nEpisode ended: {info}")
            break
    
    env.close()
    print("\nTest complete")