"""
Silksong PPO Training
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


# Safely stop

class TrainingManager:
    """Manage training status and support safely stop"""
    
    def __init__(self):
        self.should_stop = False
        self.network = None
        self.agent = None
        self.logger = None
        self.writer = None
        self.run_name = None
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Process Ctrl+C"""
        print("\n\n" + "="*60)
        print(" Stop signal received (Ctrl+C)")
        print("="*60)
        print("Saving all datas...")
        self.should_stop = True
    
    def save_checkpoint(self, update, force=False):
        """Save checkpoints"""
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
        
        print(f"Model saved: {ckpt_path}")
        return ckpt_path
    
    def cleanup(self, update):
        """Clean and save"""
        print("\nSaving...")
        
        # Save Model
        if self.network and self.agent:
            self.save_checkpoint(update, force=True)
        
        # Save log
        if self.logger:
            self.logger.save()
        
        # Close TensorBoard
        if self.writer:
            self.writer.close()
        
        print("All data saved")
        print("="*60)


# Training Manager
training_manager = TrainingManager()

# Episode Logger

class EpisodeLogger:
    """Record detailed Episode data"""
    
    def __init__(self, save_path='training_log.json'):
        self.save_path = save_path
        self.episodes = []
    
    def log_episode(self, episode_data):
        """Record an Episode"""
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
        print(f"Log saved: {self.save_path}")


# Neural Networks

class CNNEncoder(nn.Module):
    """
    CNN Feature extractor (3-layer convolution)
    Input: (batch, channels, 128, 128)
    Output: (batch, 9216)
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
        # Output Dimension：64 × 12 × 12 = 9216
        self.output_dim = 64 * 12 * 12
    
    def forward(self, x):
        return self.conv(x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network
    CNN → LSTM → Actor/Critic
    """
    def __init__(self, obs_channels=4, n_actions=10, lstm_hidden=512):
        super().__init__()
        
        # CNN Encoder
        self.encoder = CNNEncoder(obs_channels)
        
        # LSTM Memory layer
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # Actor Head
        self.actor = nn.Linear(lstm_hidden, n_actions)
        
        # Critic Head
        self.critic = nn.Linear(lstm_hidden, 1)
        
        self.lstm_hidden = lstm_hidden
    
    def forward(self, x, hidden_state=None):
        """
        x: (batch, time, channels, H, W)
        hidden_state: LSTM 
        """
        batch_size, time_steps = x.shape[:2]
        
        # CNN Feature extraction
        # (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = x.view(batch_size * time_steps, *x.shape[2:])
        features = self.encoder(x_flat)
        
        # Recovery time dimension
        # (B*T, feature_dim) -> (B, T, feature_dim)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Actor-Critic Head
        action_logits = self.actor(lstm_out)  # (B, T, n_actions)
        state_value = self.critic(lstm_out)   # (B, T, 1)
        
        return action_logits, state_value, new_hidden


# Experience Cache

class RolloutBuffer:
    """Storing a Rollout experience"""
    
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        
        # Storage space
        self.observations = torch.zeros(capacity, *obs_shape, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.log_probs = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        
        self.ptr = 0
    
    def store(self, obs, action, reward, done, log_prob, value):
        """Storage experience"""
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
        Calculate Generalized Advantage Estimation
        
        Return:
            advantages
            returns
        """
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        gae = 0
        next_value = last_value
        
        # Calculate from back to front
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
        
        # Standardization advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


# PPO Agent

class PPOAgent:
    """
    PPO Algorithm
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
        Update network
        
        Parameters:
            buffer: RolloutBuffer
            last_value
            n_epochs
            batch_size
        """
        # CompuCatational advantages and returns
        advantages, returns = buffer.compute_returns_and_advantages(last_value)
        
        # Prepare data
        obs = buffer.observations
        actions = buffer.actions
        old_log_probs = buffer.log_probs
        
        total_samples = buffer.capacity
        
        # Multiple updates
        for epoch in range(n_epochs):
            # Randomly shuffle
            indices = torch.randperm(total_samples, device=buffer.device)
            
            # Updated in batches
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Extract batch
                batch_obs = obs[batch_idx].unsqueeze(1)
                batch_actions = actions[batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                
                # Forward propagation
                logits, values, _ = self.network(batch_obs)
                logits = logits.squeeze(1)
                values = values.squeeze(1).squeeze(-1)
                
                # Calculate the new log prob
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO Loss
                # Policy loss
                ratio = (new_log_probs - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_ret)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Loss
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


# Training Function

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Automatically find the latest checkpoint
    
    Return:
        checkpoint_path or None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all .pt files
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not files:
        return None
    
    # Sort by modification time and return the latest
    files_with_time = [(f, os.path.getmtime(os.path.join(checkpoint_dir, f))) 
                       for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    latest_file = files_with_time[0][0]
    latest_path = os.path.join(checkpoint_dir, latest_file)
    
    return latest_path


def load_checkpoint(network, agent, checkpoint_path):
    """
    Load checkpoints
    
    Return:
        start_update
    """
    print(f"\nLoad Model: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Loading network parameters
    network.load_state_dict(checkpoint['network'])
    
    # Load optimizer status
    if 'optimizer' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Get the previous progress
    start_update = checkpoint.get('update', 0)
    
    print(f"Model has been loaded")
    print(f"Update {start_update}")
    
    return start_update


def warmup_exploration(env, n_episodes=50):
    """
    Pre-exploration phase
    """
    print("\n" + "="*60)
    print(f" Pre-exploration phase: {n_episodes} Episodes")
    print("="*60)
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            # Random Actions
            action = np.random.randint(0, 10)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        if (ep + 1) % 10 == 0:
            print(f"Done {ep+1}/{n_episodes} Episodes")
    
    print(f" Pre-exploration complete {n_episodes}\n")


def train(total_episodes=1000, steps_per_rollout=256, device='cuda', 
          auto_resume=True, warmup_episodes=50):
    """
    Main training function
    
    Parameter:
        total_episodes: Total number of training rounds
        steps_per_rollout: How many steps of experience to collect each time
        device: 'cuda' or 'cpu'
        auto_resume: Should previous models be loaded automatically
        warmup_episodes: Number of pre-explored Episodes
    """
    # Equipment
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Use equipment: {device}")
    
    # Creating an environment
    env = SilksongEnv(max_player_hp=6, boss_hp_max=800, frame_stack=4)
    
    # Get the observed shape
    obs_sample = env.reset()
    obs_shape = obs_sample.shape  # (4, 128, 128)
    
    print(f"Observed shape: {obs_shape}")
    network = ActorCriticNetwork(
        obs_channels=4,     # frame stack
        n_actions=10,       # Number of actions
        lstm_hidden=512     # LSTM size
    ).to(device)
    
    # Creating an Agent
    agent = PPOAgent(
        network=network,
        lr=5e-4,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.1
    )
    
    # Resume the last training session.
    start_update = 0
    if auto_resume:
        latest_ckpt = find_latest_checkpoint()
        if latest_ckpt:
            print("\n" + "="*60)
            print(f"Discover the previous model: {latest_ckpt}")
            print("="*60)
            
            choice = input("\nContinue with last training session(y/n): ").strip().lower()
            
            if choice == 'y':
                start_update = load_checkpoint(network, agent, latest_ckpt)
                print(f"From update {start_update} Continue training")
            else:
                print("Starting over")
        else:
            print("\No previous model found, starting over")
    
    # TensorBoard
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard: runs/{run_name}")
    
    # Episode Logger
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
        print(f"Continue training from Update {start_update}")
    else:
        print("Start training")
        
        # Pre-exploration
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
        # Check if it should be stopped
        if training_manager.should_stop:
            print("\Stop signal received, saving in progress...")
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
                print("\nData collection interrupted; save current data...")
                break
            
            with torch.no_grad():
                # Network output
                logits, value, hidden_state = network(obs_tensor.unsqueeze(1), hidden_state)
                logits = logits.squeeze(0).squeeze(0)
                value = value.squeeze(0).squeeze(0)
                
                # Sampling action
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Execute action
            next_obs, reward, done, info = env.step(action.item())
            
            # Storage experience
            buffer.store(
                obs_tensor.squeeze(0),
                action,
                reward,
                done,
                log_prob,
                value
            )
            
            # Update status
            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Episode ended
            if done:
                # Recorded to TensorBoard
                writer.add_scalar('train/episode_reward', episode_reward, episode_num)
                writer.add_scalar('train/episode_steps', episode_steps, episode_num)
                
                # Record Results
                result = info.get('result', 'unknown')
                if result == 'win':
                    writer.add_scalar('train/win', 1, episode_num)
                else:
                    writer.add_scalar('train/win', 0, episode_num)
                
                # Record health points
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
                
                # Record detailed data to JSON
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
                
                # Logs are saved every 50 Episodes
                if episode_num % 50 == 0 and episode_num > 0:
                    episode_logger.save()
                
                # Reset
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
                
                obs = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                hidden_state = None
        
        # Check if the collection was interrupted
        if training_manager.should_stop:
            training_manager.cleanup(update)
            return
        
        # Calculate the value in the last step
        with torch.no_grad():
            _, last_value, _ = network(obs_tensor.unsqueeze(1), hidden_state)
            last_value = last_value.squeeze()
        
        # Update networks
        losses = agent.update(buffer, last_value, n_epochs=4, batch_size=256)
        
        # Record training metrics
        writer.add_scalar('train/policy_loss', losses['policy_loss'], update)
        writer.add_scalar('train/value_loss', losses['value_loss'], update)
        writer.add_scalar('train/entropy', losses['entropy'], update)
        
        # Record every 10 updates
        if (update + 1) % 10 == 0 and len(episode_logger.episodes) > 0:
            recent_episodes = episode_logger.episodes[-50:]
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
        
        # Save model (Each 20 Updates）
        if (update + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f'checkpoints/{run_name}_u{update+1}.pt'
            torch.save({
                'network': network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'update': update + 1,
            }, ckpt_path)
            print(f"Save the model: {ckpt_path}")
    
    # Save data
    print("\nSaving in progress...")
    training_manager.cleanup(total_episodes)
    
    writer.close()
    env.close()
    
    print("\n Training complete")

# Main Function
if __name__ == '__main__':
    print("="*60)
    print("Silksong PPO Training")
    print("="*60)
    print("\nConfiguration:")
    print("  Env: SilksongEnv")
    print("  Network: CNN-LSTM + Actor-Critic")
    print("  Algorithm: PPO")
    print("  Observe: 128x128")
    print("  Action numbers: 10")
    print("  Frame stack: 4")
    print("\nTraining parameters:")
    print("  Learning rate: 5e-4")
    print("  Entropy: 0.1")
    print("  Hit reward: 2.0")
    print("  Damage punishment: 0.5")
    print("  Pre-exploration: 50 Episodes")
    print("="*60)
    print("\n Note:")
    print("  - Press Ctrl+C to stop the training")
    print("  - The model and logs will be automatically saved when stopped")
    print("  - Data stored in checkpoints/ and logs/")
    print("="*60)
    
    input("\nMake sure in the combat position, then press Enter to begin training....")
    
    # Start train
    try:
        train(
            total_episodes=100,      # 100 Updates
            steps_per_rollout=128,   # 128 steps
            device='cpu',            # Use CPU
            auto_resume=True,        # Automatically continue from the last training
            warmup_episodes=50       # Pre-exploration 50 Episodes
        )
        
    # Print status
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTerminate")
