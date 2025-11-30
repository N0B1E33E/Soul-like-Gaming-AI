"""
Silksong PPO 训练脚本
基于你们原有的 PPO 实现，适配新环境

按 Ctrl+C 可以安全停止并保存所有数据
"""

import os
import time
import json
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from silksong_env import SilksongEnv


# ============================================
# 安全停止处理
# ============================================

class TrainingManager:
    """管理训练状态，支持安全停止"""
    
    def __init__(self):
        self.should_stop = False
        self.network = None
        self.agent = None
        self.logger = None
        self.writer = None
        self.run_name = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """处理 Ctrl+C"""
        print("\n\n" + "="*60)
        print(" 收到停止信号 (Ctrl+C)")
        print("="*60)
        print("正在安全保存所有数据...")
        self.should_stop = True
    
    def save_checkpoint(self, update, force=False):
        """保存检查点"""
        if self.network is None or self.agent is None:
            return
        
        os.makedirs('checkpoints', exist_ok=True)
        
        if force:
            ckpt_path = f'checkpoints/{self.run_name}_interrupted_u{update}.pt'
        else:
            ckpt_path = f'checkpoints/{self.run_name}_u{update}.pt'
        
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.agent.optimizer.state_dict(),
            'update': update,
            'run_name': self.run_name,
        }, ckpt_path)
        
        print(f" 模型已保存: {ckpt_path}")
        return ckpt_path
    
    def cleanup(self, update):
        """清理和保存"""
        print("\n正在保存...")
        
        # 保存模型
        if self.network and self.agent:
            self.save_checkpoint(update, force=True)
        
        # 保存日志
        if self.logger:
            self.logger.save()
        
        # 关闭 TensorBoard
        if self.writer:
            self.writer.close()
        
        print(" 所有数据已保存")
        print("="*60)


# 全局训练管理器
training_manager = TrainingManager()


# ============================================
# Episode 日志记录器（可选）
# ============================================

class EpisodeLogger:
    """记录详细的 Episode 数据"""
    
    def __init__(self, save_path='training_log.json'):
        self.save_path = save_path
        self.episodes = []
    
    def log_episode(self, episode_data):
        """记录一个 Episode"""
        self.episodes.append(episode_data)
    
    def save(self):
        """保存到文件"""
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_episodes': len(self.episodes),
            'episodes': self.episodes
        }
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✓ 日志已保存: {self.save_path}")


# ============================================
# 第1部分：神经网络
# ============================================

class CNNEncoder(nn.Module):
    """
    CNN 特征提取器（3层卷积）
    输入：(batch, channels, 128, 128)
    输出：(batch, 9216) 扁平化特征
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 128->31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 31->14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 14->12
            nn.ReLU(),
            nn.Flatten()
        )
        # 输出维度：64 × 12 × 12 = 9216
        self.output_dim = 64 * 12 * 12
    
    def forward(self, x):
        return self.conv(x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    CNN → LSTM → Actor/Critic 双头
    """
    def __init__(self, obs_channels=4, n_actions=10, lstm_hidden=512):
        super().__init__()
        
        # CNN 编码器
        self.encoder = CNNEncoder(obs_channels)
        
        # LSTM 记忆层
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # Actor 头（输出动作概率）
        self.actor = nn.Linear(lstm_hidden, n_actions)
        
        # Critic 头（输出状态价值）
        self.critic = nn.Linear(lstm_hidden, 1)
        
        self.lstm_hidden = lstm_hidden
    
    def forward(self, x, hidden_state=None):
        """
        x: (batch, time, channels, H, W)
        hidden_state: LSTM 隐藏状态
        """
        batch_size, time_steps = x.shape[:2]
        
        # CNN 提取特征
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = x.view(batch_size * time_steps, *x.shape[2:])
        features = self.encoder(x_flat)
        
        # 恢复时间维度
        # (B*T, feature_dim) -> (B, T, feature_dim)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Actor-Critic 头
        action_logits = self.actor(lstm_out)  # (B, T, n_actions)
        state_value = self.critic(lstm_out)   # (B, T, 1)
        
        return action_logits, state_value, new_hidden


# ============================================
# 第2部分：经验缓存
# ============================================

class RolloutBuffer:
    """存储一个 Rollout 的经验"""
    
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        
        # 存储空间
        self.observations = torch.zeros(capacity, *obs_shape, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        
        self.ptr = 0
    
    def store(self, obs, action, reward, done, log_prob, value):
        """存储一步经验"""
        idx = self.ptr % self.capacity
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        计算 GAE（Generalized Advantage Estimation）
        
        返回：
            advantages: 优势函数
            returns: 回报
        """
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        next_value = last_value
        
        # 从后往前计算
        for t in reversed(range(self.capacity)):
            mask = 1.0 - self.dones[t]
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            
            # GAE
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            
            next_value = self.values[t]
        
        # Returns
        returns = advantages + self.values
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


# ============================================
# 第3部分：PPO Agent
# ============================================

class PPOAgent:
    """
    PPO 算法实现
    """
    
    def __init__(self, network, lr=2.5e-4, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # PPO 参数
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, buffer, last_value, n_epochs=4, batch_size=256):
        """
        更新网络
        
        参数：
            buffer: RolloutBuffer
            last_value: 最后一步的状态价值
            n_epochs: 更新轮数
            batch_size: 批次大小
        """
        # 计算优势和回报
        advantages, returns = buffer.compute_returns_and_advantages(last_value)
        
        # 准备数据
        obs = buffer.observations
        actions = buffer.actions
        old_log_probs = buffer.log_probs
        
        total_samples = buffer.capacity
        
        # 多轮更新
        for epoch in range(n_epochs):
            # 随机打乱
            indices = torch.randperm(total_samples, device=buffer.device)
            
            # 分批更新
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # 提取 batch
                batch_obs = obs[batch_idx].unsqueeze(1)  # 加时间维度
                batch_actions = actions[batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                
                # 前向传播
                logits, values, _ = self.network(batch_obs)
                logits = logits.squeeze(1)  # 去掉时间维度
                values = values.squeeze(1).squeeze(-1)
                
                # 计算新的 log prob
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO Loss
                # 1. Policy loss（带 clip）
                ratio = (new_log_probs - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Value loss
                value_loss = F.mse_loss(values, batch_ret)
                
                # 3. Entropy bonus（鼓励探索）
                entropy = dist.entropy().mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


# ============================================
# 第4部分：训练函数
# ============================================

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    自动寻找最新的检查点
    
    返回：
        checkpoint_path: 最新模型的路径
        或 None（如果没有）
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 找所有 .pt 文件
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not files:
        return None
    
    # 按修改时间排序，返回最新的
    files_with_time = [(f, os.path.getmtime(os.path.join(checkpoint_dir, f))) 
                       for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_file = files_with_time[0][0]
    latest_path = os.path.join(checkpoint_dir, latest_file)
    
    return latest_path


def load_checkpoint(network, agent, checkpoint_path):
    """
    加载检查点
    
    返回：
        start_update: 从哪个 update 继续
    """
    print(f"\n加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载网络参数
    network.load_state_dict(checkpoint['network'])
    
    # 加载优化器状态
    if 'optimizer' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 获取上次的进度
    start_update = checkpoint.get('update', 0)
    
    print(f" 模型已加载")
    print(f"   上次训练到: Update {start_update}")
    
    return start_update


def warmup_exploration(env, n_episodes=50):
    """
    预探索阶段（参考 seermer）
    
    随机玩几个 Episodes，让 Agent 见识各种情况
    不训练，只是"体验"游戏
    """
    print("\n" + "="*60)
    print(f" 预探索阶段: {n_episodes} Episodes")
    print("="*60)
    print("随机动作，不训练，只是让 AI 体验游戏")
    print("这样能学到：")
    print("  - 各种动作的效果")
    print("  - Boss 的攻击模式")
    print("  - 什么情况会死")
    print("="*60)
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            # 随机动作
            action = np.random.randint(0, 10)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  完成 {ep+1}/{n_episodes} Episodes")
    
    print(f" 预探索完成！体验了 {n_episodes} 个 Episodes\n")


def train(total_episodes=1000, steps_per_rollout=256, device='cuda', 
          auto_resume=True, warmup_episodes=50):
    """
    主训练函数
    
    参数：
        total_episodes: 总训练轮数
        steps_per_rollout: 每次收集多少步经验
        device: 'cuda' 或 'cpu'
        auto_resume: 是否自动加载之前的模型
        warmup_episodes: 预探索 Episodes 数量
    """
    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建环境
    env = SilksongEnv(max_player_hp=6, boss_hp_max=800, frame_stack=4)
    
    # 获取观察形状
    obs_sample = env.reset()
    obs_shape = obs_sample.shape  # (4, 128, 128)
    
    print(f"观察形状: {obs_shape}")
    network = ActorCriticNetwork(
        obs_channels=4,     # frame stack
        n_actions=10,       # 动作数
        lstm_hidden=512     # LSTM 大小
    ).to(device)
    
    # 创建 Agent（激进参数 - 快速学习）
    agent = PPOAgent(
        network=network,
        lr=5e-4,               # 学习率加倍（快速学习）
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.1       # 充分探索
    )
    
    # 自动恢复上次的训练
    start_update = 0
    if auto_resume:
        latest_ckpt = find_latest_checkpoint()
        if latest_ckpt:
            print("\n" + "="*60)
            print(f"发现之前的模型: {latest_ckpt}")
            print("="*60)
            
            choice = input("\n要继续上次的训练吗？(y/n): ").strip().lower()
            
            if choice == 'y':
                start_update = load_checkpoint(network, agent, latest_ckpt)
                print(f"将从 Update {start_update} 继续训练")
            else:
                print("从零开始新的训练")
        else:
            print("\n没有找到之前的模型，从零开始")
    
    # TensorBoard
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard: runs/{run_name}")
    
    # Episode Logger（详细数据）
    episode_logger = EpisodeLogger(f'logs/training_{run_name}.json')
    os.makedirs('logs', exist_ok=True)
    
    # 注册到训练管理器
    training_manager.network = network
    training_manager.agent = agent
    training_manager.logger = episode_logger
    training_manager.writer = writer
    training_manager.run_name = run_name
    
    # 训练循环
    print("\n" + "="*60)
    if start_update > 0:
        print(f"继续训练（从 Update {start_update} 开始）")
    else:
        print("开始训练（从零开始）")
        
        # 预探索（只在从零开始时）
        if warmup_episodes > 0:
            warmup_exploration(env, warmup_episodes)
    
    print("="*60)
    
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    hidden_state = None
    
    episode_num = 0
    episode_reward = 0
    episode_steps = 0
    
    global_step = 0
    
    for update in range(start_update, total_episodes):
        # 检查是否应该停止
        if training_manager.should_stop:
            print("\n收到停止信号，正在保存...")
            training_manager.cleanup(update)
            return
        
        # 创建 buffer
        buffer = RolloutBuffer(
            capacity=steps_per_rollout,
            obs_shape=obs.shape,
            device=device
        )
        
        # 收集经验
        for t in range(steps_per_rollout):
            # 检查停止信号
            if training_manager.should_stop:
                print("\n收集中断，保存当前数据...")
                break
            
            with torch.no_grad():
                # 网络输出
                logits, value, hidden_state = network(obs_tensor.unsqueeze(1), hidden_state)
                logits = logits.squeeze(0).squeeze(0)
                value = value.squeeze(0).squeeze(0)
                
                # 采样动作
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action.item())
            
            # 存储经验
            buffer.store(
                obs_tensor.squeeze(0),
                action,
                reward,
                done,
                log_prob,
                value
            )
            
            # 更新状态
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Episode 结束
            if done:
                # 记录到 TensorBoard
                writer.add_scalar('train/episode_reward', episode_reward, episode_num)
                writer.add_scalar('train/episode_steps', episode_steps, episode_num)
                
                # 记录结果
                result = info.get('result', 'unknown')
                if result == 'win':
                    writer.add_scalar('train/win', 1, episode_num)
                else:
                    writer.add_scalar('train/win', 0, episode_num)
                
                # 记录血量信息
                final_player_hp = info.get('player_hp', 0)
                final_boss_hp = info.get('boss_hp', 800)
                writer.add_scalar('train/final_player_hp', final_player_hp, episode_num)
                writer.add_scalar('train/final_boss_hp', final_boss_hp, episode_num)
                writer.add_scalar('train/boss_damage', 800 - final_boss_hp, episode_num)
                
                print(f"Episode {episode_num}: "
                      f"Steps={episode_steps}, "
                      f"Reward={episode_reward:.2f}, "
                      f"Result={result}, "
                      f"Player={final_player_hp}/6, "
                      f"Boss={final_boss_hp:.0f}/800")
                
                # 记录详细数据到 JSON
                episode_logger.log_episode({
                    'episode': episode_num,
                    'steps': episode_steps,
                    'reward': round(episode_reward, 2),
                    'result': result,
                    'final_player_hp': final_player_hp,
                    'final_boss_hp': round(final_boss_hp, 1),
                    'boss_damage': round(800 - final_boss_hp, 1),
                    'boss_damage_percent': round((800 - final_boss_hp) / 800 * 100, 1),
                    'survival_rate': round(final_player_hp / 6 * 100, 1),
                    'global_step': global_step,
                    'update': update
                })
                
                # 每50个 Episode 保存一次日志
                if episode_num % 50 == 0 and episode_num > 0:
                    episode_logger.save()
                
                # 重置
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
                
                obs = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hidden_state = None
        
        # 检查是否在收集中被中断
        if training_manager.should_stop:
            training_manager.cleanup(update)
            return
        
        # 计算最后一步的 value
        with torch.no_grad():
            _, last_value, _ = network(obs_tensor.unsqueeze(1), hidden_state)
            last_value = last_value.squeeze()
        
        # 更新网络
        losses = agent.update(buffer, last_value, n_epochs=4, batch_size=256)
        
        # 记录训练指标
        writer.add_scalar('train/policy_loss', losses['policy_loss'], update)
        writer.add_scalar('train/value_loss', losses['value_loss'], update)
        writer.add_scalar('train/entropy', losses['entropy'], update)
        
        # 每10次 update 记录统计数据
        if (update + 1) % 10 == 0 and len(episode_logger.episodes) > 0:
            recent_episodes = episode_logger.episodes[-50:]  # 最近50个
            if recent_episodes:
                avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
                avg_reward = np.mean([ep['reward'] for ep in recent_episodes])
                avg_boss_dmg = np.mean([ep['boss_damage'] for ep in recent_episodes])
                win_rate = sum(1 for ep in recent_episodes if ep['result'] == 'win') / len(recent_episodes)
                
                writer.add_scalar('stats/avg_episode_steps', avg_steps, update)
                writer.add_scalar('stats/avg_episode_reward', avg_reward, update)
                writer.add_scalar('stats/avg_boss_damage', avg_boss_dmg, update)
                writer.add_scalar('stats/win_rate', win_rate, update)
        
        # 打印
        if (update + 1) % 10 == 0:
            print(f"\n[Update {update+1}/{total_episodes}]")
            print(f"  Policy Loss: {losses['policy_loss']:.4f}")
            print(f"  Value Loss: {losses['value_loss']:.4f}")
            print(f"  Entropy: {losses['entropy']:.4f}\n")
        
        # 保存模型（每 20 Updates）
        if (update + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f'checkpoints/{run_name}_u{update+1}.pt'
            torch.save({
                'network': network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'update': update + 1,
            }, ckpt_path)
            print(f"✓ 保存模型: {ckpt_path}")
    
    # 正常结束，保存数据
    print("\n训练完成，正在保存...")
    training_manager.cleanup(total_episodes)
    
    writer.close()
    env.close()
    
    print("\n 训练完成！")


# ============================================
# 主程序
# ============================================

if __name__ == '__main__':
    print("="*60)
    print("Silksong PPO 训练 - 激进版")
    print("="*60)
    print("\n配置:")
    print("  环境: SilksongEnv")
    print("  网络: CNN-LSTM + Actor-Critic")
    print("  算法: PPO")
    print("  观察: 128×128 (更多信息)")
    print("  动作数: 10")
    print("  Frame stack: 4")
    print("\n训练参数（激进 - 快速学习）:")
    print("  学习率: 5e-4 (加倍)")
    print("  Entropy: 0.1 (更多探索)")
    print("  击中奖励: 2.0 (大力鼓励)")
    print("  受伤惩罚: 0.5 (减少恐惧)")
    print("  预探索: 50 Episodes")
    print("="*60)
    print("\n 重要提示:")
    print("  - 按 Ctrl+C 可以安全停止训练")
    print("  - 停止时会自动保存模型和日志")
    print("  - 数据保存在 checkpoints/ 和 logs/")
    print("="*60)
    
    input("\n确保游戏在战斗位置，按 Enter 开始训练...")
    
    # 开始训练
    try:
        train(
            total_episodes=100,      # 100 Updates
            steps_per_rollout=128,   # 128 步（减少 OCR 调用）
            device='cpu',            # 使用 CPU
            auto_resume=True,        # 自动继续上次训练
            warmup_episodes=50       # 预探索 50 Episodes
        )
        
        # 时间估算：约 4-5 小时
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n程序退出")