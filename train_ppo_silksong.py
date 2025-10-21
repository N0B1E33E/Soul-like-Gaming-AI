import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from game_env import GameEnv, ACTION_SPACE


# ============== 模型 ==============
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.out_dim = 64 * 7 * 7  # 84x84 -> 7x7

    def forward(self, x):
        return self.net(x)


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_channels=1, n_actions=12, lstm_hidden=512):
        super().__init__()
        self.enc = CNNEncoder(obs_channels)
        self.lstm = nn.LSTM(self.enc.out_dim, lstm_hidden, batch_first=True)
        self.actor = nn.Linear(lstm_hidden, n_actions)
        self.critic = nn.Linear(lstm_hidden, 1)

    def forward(self, x, h=None):
        # x: [B,T,C,H,W]
        B, T = x.shape[:2]
        z = self.enc(x.view(B*T, *x.shape[2:])).view(B, T, -1)
        z, h = self.lstm(z, h)
        logits = self.actor(z)   # [B,T,A]
        value  = self.critic(z)  # [B,T,1]
        return logits, value, h


# ============== Rollout Buffer ==============
class RolloutBuffer:
    def __init__(self, T, B, obs_shape, device):
        self.T, self.B = T, B
        self.device = device
        self.obs = torch.zeros(T, B, *obs_shape, device=device)
        self.act = torch.zeros(T, B, dtype=torch.long, device=device)
        self.rew = torch.zeros(T, B, device=device)
        self.done= torch.zeros(T, B, device=device)
        self.logp= torch.zeros(T, B, device=device)
        self.val = torch.zeros(T, B, device=device)

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        adv = torch.zeros_like(self.rew)
        gae = torch.zeros(self.B, device=self.rew.device)
        for t in reversed(range(self.T)):
            mask = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * last_val * mask - self.val[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            last_val = self.val[t]
        ret = adv + self.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret


# ============== PPO ==============
class PPO:
    def __init__(self, model, lr=2.5e-4, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.opt = optim.Adam(model.parameters(), lr=lr)
        self.clip = clip; self.vf_coef = vf_coef; self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def update(self, obs, act, logp_old, adv, ret, epochs=4, batch_size=1024):
        B, T = obs.shape[0], obs.shape[1]
        N = B * T
        flat_obs = obs.view(N, *obs.shape[2:])
        flat_act = act.view(N)
        flat_logp_old = logp_old.view(N)
        flat_adv = adv.view(N)
        flat_ret = ret.view(N)

        idx_full = torch.arange(N, device=obs.device)
        for _ in range(epochs):
            perm = idx_full[torch.randperm(N)]
            for i in range(0, N, batch_size):
                j = perm[i:i+batch_size]
                x = flat_obs[j].unsqueeze(1)  # [b,1,C,H,W]
                logits, value, _ = self.model(x)
                value = value.squeeze(1).squeeze(-1)         # [b]
                dist = Categorical(logits=logits.squeeze(1)) # [b,A]
                logp = dist.log_prob(flat_act[j])            # [b]
                ratio = (logp - flat_logp_old[j]).exp()

                surr1 = ratio * flat_adv[j]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * flat_adv[j]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.smooth_l1_loss(value, flat_ret[j])
                entropy     = dist.entropy().mean()
                loss = policy_loss + self.vf_coef*value_loss - self.ent_coef*entropy

                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()


# ============== 评测 ==============
def evaluate(env, model, device, episodes=5):
    model.eval()
    wins, steps = 0, []
    with torch.no_grad():
        for _ in range(episodes):
            obs = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,H,W]
            h = None; t = 0; done = False; info = {}
            while not done and t < env.max_steps:
                logits, _, h = model(obs.unsqueeze(1), h)  # [1,1,...]
                action = torch.argmax(logits.squeeze(0).squeeze(0), dim=-1)
                obs_, _, done, info = env.step(action.item())
                obs = torch.tensor(obs_, dtype=torch.float32, device=device).unsqueeze(0)
                t += 1
            steps.append(t)
            if info.get("result") == "boss_down":
                wins += 1
    model.train()
    return {"win_rate": wins/episodes, "mean_steps": float(np.mean(steps))}


# ============== 训练主函数 ==============
def train(total_updates=500, T=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = GameEnv()
    obs_np = env.reset()                    # obs_np.shape == (C,84,84) ，此处 C=4
    obs_channels = obs_np.shape[0]
    obs_shape = (obs_channels, 84, 84)
    n_actions = len(ACTION_SPACE)
    model = RecurrentActorCritic(obs_channels=obs_channels, n_actions=n_actions).to(device)
    ppo = PPO(model)

    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,84,84]
    h = None
    
    # TensorBoard
    run_name = time.strftime("silksong_%Y%m%d-%H%M%S")
    logdir = os.path.join("runs", run_name)
    writer = SummaryWriter(logdir)
    print(f"[TB] logging to {logdir}")

    # 采样/训练循环
    obs_np = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)  # [1,C,H,W]
    h = None

    for update in range(total_updates):
        buf = RolloutBuffer(T, 1, obs_shape, device)

        ep_rewards = []
        ep_reward = 0.0

        for t in range(T):
            with torch.no_grad():
                logits, value, h = model(obs.unsqueeze(1), h)  # [1,1,...]
                dist = Categorical(logits=logits.squeeze(0).squeeze(0))
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            buf.obs[t,0] = obs[0]
            buf.act[t,0] = action
            buf.rew[t,0] = reward
            buf.done[t,0] = float(done)
            buf.logp[t,0] = logp
            buf.val[t,0]  = value.squeeze()

            obs = next_obs_t
            ep_reward += reward

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                h = None
                obs = torch.tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            last_v = model(obs.unsqueeze(1), h)[1].squeeze()
        adv, ret = buf.compute_gae(last_v)

        # PPO 更新
        ppo.update(buf.obs, buf.act, buf.logp, adv, ret, epochs=4, batch_size=512)

        # TB 日志
        mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else float(buf.rew.sum().item())
        writer.add_scalar("train/mean_ep_reward", mean_ep_reward, update)
        writer.add_scalar("train/adv_std", adv.std().item(), update)
        writer.add_scalar("train/adv_mean", adv.mean().item(), update)

        # 每20次：保存 + 评测
        if (update + 1) % 20 == 0:
            # 保存 ckpt
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"{run_name}_u{update+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "opt": ppo.opt.state_dict(),
                "update": update + 1,
                "run_name": run_name,
            }, ckpt_path)
            print(f"[CKPT] saved -> {ckpt_path}")

            # 评测
            stats = evaluate(env, model, device, episodes=5)
            writer.add_scalar("eval/win_rate",  stats["win_rate"], update)
            writer.add_scalar("eval/mean_steps", stats["mean_steps"], update)

        if (update + 1) % 5 == 0:
            print(f"[Update {update+1}] mean_ep_reward={mean_ep_reward:.4f} adv_std={adv.std().item():.3f}")

    writer.close()


if __name__ == "__main__":
    train(total_updates=500, T=128)
