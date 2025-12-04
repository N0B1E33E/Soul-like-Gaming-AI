"""
Silksong PPO Training script
Based on your existing PPO implementation, adapted for the new environment

Press Ctrl+C to safely stop and save all data.

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
# Safety Stop Handling
# ============================================

class TrainingManager:
    """Manages training state, supports safe stopping"""

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
        """Handle Ctrl+C"""
        print("\n\n" + "="*60)
        print(" Received stop signal (Ctrl+C)")
        print("="*60)
        print("Safely saving all data...")
        self.should_stop = True
    
    def save_checkpoint(self, update, force=False):
        """Save checkpoint"""
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
        
        print(f" Model saved: {ckpt_path}")
        return ckpt_path
    
    def cleanup(self, update):
        """Cleaning and saving"""
        print("\nSaving...")
        
        # Save model
        if self.network and self.agent:
            self.save_checkpoint(update, force=True)
        
        # Save logs
        if self.logger:
            self.logger.save()
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        print(" All data saved")
        print("="*60)


# Global training manager
training_manager = TrainingManager()


# ============================================
# Episode Logger (Optional)
# ============================================

class EpisodeLogger:
    """Logs detailed episode data"""
    
    def __init__(self, save_path='training_log.json'):
        self.save_path = save_path
        self.episodes = []
    
    def log_episode(self, episode_data):
        """Log an episode"""
        self.episodes.append(episode_data)
    
    def save(self):
        """Save to file"""
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_episodes': len(self.episodes),
            'episodes': self.episodes
        }
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Log saved: {self.save_path}")


# ============================================
# Part 1: Neural Network
# ============================================

class CNNEncoder(nn.Module):
    """
    CNN Feature Extractor (3 convolutional layers)
    Input: (batch, channels, 128, 128)
    Output: (batch, 9216) Flattened features
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
        # Output dimension: 64 × 12 × 12 = 9216
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
        
        # CNN Encoder
        self.encoder = CNNEncoder(obs_channels)
        
        # LSTM Memory Layer
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # Actor head (outputs action probabilities)
        self.actor = nn.Linear(lstm_hidden, n_actions)
        
        # Critic head (outputs state value)
        self.critic = nn.Linear(lstm_hidden, 1)
        
        self.lstm_hidden = lstm_hidden
    
    def forward(self, x, hidden_state=None):
        """
        x: (batch, time, channels, H, W)
        hidden_state: LSTM hidden state
        """
        batch_size, time_steps = x.shape[:2]
        
        # CNN feature extraction
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = x.view(batch_size * time_steps, *x.shape[2:])
        features = self.encoder(x_flat)
        
        # Restore time dimension
        # (B*T, feature_dim) -> (B, T, feature_dim)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Actor-Critic heads
        action_logits = self.actor(lstm_out)  # (B, T, n_actions)
        state_value = self.critic(lstm_out)   # (B, T, 1)
        
        return action_logits, state_value, new_hidden


# ============================================
# Part 2: Experience Buffer
# ============================================

class RolloutBuffer:
    """Stores experiences for one rollout"""
    
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        
        # Storage
        self.observations = torch.zeros(capacity, *obs_shape, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        
        self.ptr = 0
    
    def store(self, obs, action, reward, done, log_prob, value):
        """Store one step of experience"""
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
        Compute GAE (Generalized Advantage Estimation)
        
        Returns:
            advantages: Advantages
            returns: Returns
        """
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        next_value = last_value
        
        # Compute from the end to the beginning
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
        
        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


# ============================================
# Part 3: PPO Agent
# ============================================

class PPOAgent:
    """
    PPO algorithm implementation
    """
    
    def __init__(self, network, lr=2.5e-4, clip_ratio=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # PPO parameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, buffer, last_value, n_epochs=4, batch_size=256):
        """
        Update the network
        
        Parameters:
            buffer: RolloutBuffer
            last_value: Last state's value
            n_epochs: Number of update epochs
            batch_size: Batch size
        """
        # Compute advantages and returns
        advantages, returns = buffer.compute_returns_and_advantages(last_value)
        
        # Prepare data
        obs = buffer.observations
        actions = buffer.actions
        old_log_probs = buffer.log_probs
        
        total_samples = buffer.capacity
        
        # Multiple update epochs
        for epoch in range(n_epochs):
            # Random shuffle
            indices = torch.randperm(total_samples, device=buffer.device)
            
            # Batch update
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Extract batch
                batch_obs = obs[batch_idx].unsqueeze(1)  # Add time dimension
                batch_actions = actions[batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                
                # Forward pass
                logits, values, _ = self.network(batch_obs)
                logits = logits.squeeze(1)  # Remove time dimension
                values = values.squeeze(1).squeeze(-1)
                
                # Compute new log prob
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO Loss
                # 1. Policy loss (with clip)
                ratio = (new_log_probs - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Value loss
                value_loss = F.mse_loss(values, batch_ret)
                
                # 3. Entropy bonus (encourage exploration)
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backpropagation
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
# Part 4: Training Functions
# ============================================

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Automatically find the latest checkpoint
    
    Returns:
        checkpoint_path: Path to the latest checkpoint
        or None (if none found)
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all .pt files
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not files:
        return None
    
    # Sort by modification time, return the latest
    files_with_time = [(f, os.path.getmtime(os.path.join(checkpoint_dir, f))) 
                       for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_file = files_with_time[0][0]
    latest_path = os.path.join(checkpoint_dir, latest_file)
    
    return latest_path


def load_checkpoint(network, agent, checkpoint_path):
    """
    Load checkpoint
    
    Returns:
        start_update: Which update to continue from
    """
    print(f"\nLoading model: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load network parameters
    network.load_state_dict(checkpoint['network'])
    
    # Load optimizer state
    if 'optimizer' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Get last progress
    start_update = checkpoint.get('update', 0)
    
    print(f" Model loaded successfully")
    print(f"   Last trained update: {start_update}")
    
    return start_update


def warmup_exploration(env, n_episodes=50):
    """
    Warm-up exploration phase (inspired by seermer)
    
    Randomly play several episodes to let the agent experience various situations.
    No training, just "experiencing" the game.
    """
    print("\n" + "="*60)
    print(f" Warm-up phase: {n_episodes} Episodes")
    print("="*60)
    print("Random actions, no training, just letting the AI experience the game")
    print("This helps the AI learn:")
    print("  - Effects of various actions")
    print("  - Boss's attack patterns")
    print("  - Situations that lead to death")
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
            print(f"  Completed {ep+1}/{n_episodes} Episodes")
    
    print(f" Warm-up completed! Experienced {n_episodes} Episodes\n")


def train(total_episodes=1000, steps_per_rollout=256, device='cuda', 
          auto_resume=True, warmup_episodes=50):
    """
    Main training function
    
    Args:
        total_episodes: Total training episodes
        steps_per_rollout: How many steps of experience to collect each rollout
        device: 'cuda' or 'cpu'
        auto_resume: Whether to automatically load the previous model
        warmup_episodes: Number of warm-up exploration episodes
    """
    # Device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = SilksongEnv(max_player_hp=6, boss_hp_max=800, frame_stack=4)
    
    # Get observation shape
    obs_sample = env.reset()
    obs_shape = obs_sample.shape  # (4, 128, 128)
    
    print(f"Observation shape: {obs_shape}")
    network = ActorCriticNetwork(
        obs_channels=4,     # frame stack
        n_actions=10,       # Number of actions
        lstm_hidden=512     # LSTM size
    ).to(device)
    
    # Create Agent (aggressive parameters - fast learning)
    agent = PPOAgent(
        network=network,
        lr=5e-4,               # Learning rate doubled (fast learning)
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.1       # Encourage exploration
    )
    
    # Automatically resume previous training
    start_update = 0
    if auto_resume:
        latest_ckpt = find_latest_checkpoint()
        if latest_ckpt:
            print("\n" + "="*60)
            print(f"Found previous model: {latest_ckpt}")
            print("="*60)
            
            choice = input("\nContinue training from last time? (y/n): ").strip().lower()
            
            if choice == 'y':
                start_update = load_checkpoint(network, agent, latest_ckpt)
                print(f"Continuing training from Update {start_update}")
            else:
                print("Starting new training from scratch")
        else:
            print("\nNo previous model found, starting from scratch")
    
    # TensorBoard
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard: runs/{run_name}")
    
    # Episode Logger (detailed data)
    episode_logger = EpisodeLogger(f'logs/training_{run_name}.json')
    os.makedirs('logs', exist_ok=True)
    
    # Register to training manager
    training_manager.network = network
    training_manager.agent = agent
    training_manager.logger = episode_logger
    training_manager.writer = writer
    training_manager.run_name = run_name
    
    # Training loop
    print("\n" + "="*60)
    if start_update > 0:
        print(f"Continuing training from Update {start_update}")
    else:
        print("Starting training from scratch")
        
        # Warm-up exploration (only when starting from scratch)
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
        # Check if should stop
        if training_manager.should_stop:
            print("\nReceived stop signal, saving...")
            training_manager.cleanup(update)
            return
        
        # Create buffer
        buffer = RolloutBuffer(
            capacity=steps_per_rollout,
            obs_shape=obs.shape,
            device=device
        )
        
        # Collect experience
        for t in range(steps_per_rollout):
            # Check stop signal
            if training_manager.should_stop:
                print("\nCollection interrupted, saving current data...")
                break
            
            with torch.no_grad():
                # Network output
                logits, value, hidden_state = network(obs_tensor.unsqueeze(1), hidden_state)
                logits = logits.squeeze(0).squeeze(0)
                value = value.squeeze(0).squeeze(0)
                
                # Sample action
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Execute action
            next_obs, reward, done, info = env.step(action.item())
            
            # Store experience  
            buffer.store(
                obs_tensor.squeeze(0),
                action,
                reward,
                done,
                log_prob,
                value
            )
            
            # Update state
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Episode ends
            if done:
                # Log to TensorBoard
                writer.add_scalar('train/episode_reward', episode_reward, episode_num)
                writer.add_scalar('train/episode_steps', episode_steps, episode_num)
                
                # Log result
                result = info.get('result', 'unknown')
                if result == 'win':
                    writer.add_scalar('train/win', 1, episode_num)
                else:
                    writer.add_scalar('train/win', 0, episode_num)
                
                # Log health information
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
                
                # Log detailed data to JSON
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
                
                # Save logs every 50 episodes
                if episode_num % 50 == 0 and episode_num > 0:
                    episode_logger.save()
                
                # Reset for next episode
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
                
                obs = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hidden_state = None
        
        # Check if collection was interrupted
        if training_manager.should_stop:
            training_manager.cleanup(update)
            return
        
        # Calculate value for the last step
        with torch.no_grad():
            _, last_value, _ = network(obs_tensor.unsqueeze(1), hidden_state)
            last_value = last_value.squeeze()
        
        # Update network
        losses = agent.update(buffer, last_value, n_epochs=4, batch_size=256)
        
        # Log training metrics
        writer.add_scalar('train/policy_loss', losses['policy_loss'], update)
        writer.add_scalar('train/value_loss', losses['value_loss'], update)
        writer.add_scalar('train/entropy', losses['entropy'], update)
        
        # Log statistics every 10 updates
        if (update + 1) % 10 == 0 and len(episode_logger.episodes) > 0:
            recent_episodes = episode_logger.episodes[-50:]  # Last 50 episodes
            if recent_episodes:
                avg_steps = np.mean([ep['steps'] for ep in recent_episodes])
                avg_reward = np.mean([ep['reward'] for ep in recent_episodes])
                avg_boss_dmg = np.mean([ep['boss_damage'] for ep in recent_episodes])
                win_rate = sum(1 for ep in recent_episodes if ep['result'] == 'win') / len(recent_episodes)
                
                writer.add_scalar('stats/avg_episode_steps', avg_steps, update)
                writer.add_scalar('stats/avg_episode_reward', avg_reward, update)
                writer.add_scalar('stats/avg_boss_damage', avg_boss_dmg, update)
                writer.add_scalar('stats/win_rate', win_rate, update)
        
        # Print
        if (update + 1) % 10 == 0:
            print(f"\n[Update {update+1}/{total_episodes}]")
            print(f"  Policy Loss: {losses['policy_loss']:.4f}")
            print(f"  Value Loss: {losses['value_loss']:.4f}")
            print(f"  Entropy: {losses['entropy']:.4f}\n")
        
        # Save model (every 20 updates)
        if (update + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f'checkpoints/{run_name}_u{update+1}.pt'
            torch.save({
                'network': network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'update': update + 1,
            }, ckpt_path)
            print(f"✓ Saved model: {ckpt_path}")
    
    # Normal completion, save data
    print("\nTraining complete, saving...")
    training_manager.cleanup(total_episodes)
    
    writer.close()
    env.close()
    
    print("\nTraining complete!")


# ============================================
# Main program
# ============================================

if __name__ == '__main__':
    print("="*60)
    print("Silksong PPO Training - Aggressive Version")
    print("="*60)
    print("\nConfiguration:")
    print("  Environment: SilksongEnv")
    print("  Network: CNN-LSTM + Actor-Critic")
    print("  Algorithm: PPO")
    print("  Observation: 128×128 (more information)")
    print("  Number of actions: 10")
    print("  Frame stack: 4")
    print("\nTraining parameters (Aggressive - Fast Learning):")
    print("  Learning rate: 5e-4 (doubled)")
    print("  Entropy: 0.1 (More exploration)")
    print("  Hit reward: 2.0 (Added to encourage damage)")
    print("  Increase punishment: 0.5 (Reduce fear)")
    print("  Warmup: 50 Episodes")
    print("="*60)
    print("\n Important notes:")
    print("  - Ctrl+C can safely stop training at any time")
    print("  - Models and logs are automatically saved on stop")
    print("  - Data is saved in checkpoints/ and logs/")
    print("="*60)
    
    input("\nMake sure the game is in the battle position, press Enter to start training...")
    
    # Start training
    try:
        train(
            total_episodes=100,      # 100 Updates
            steps_per_rollout=128,   # 128 steps (reduce OCR calls)
            device='cpu',            # Use CPU
            auto_resume=True,        # Auto resume previous training
            warmup_episodes=50       # Warmup 50 Episodes
        )
        
        # Estimated time: about 4-5 hours
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProgram exited.")
