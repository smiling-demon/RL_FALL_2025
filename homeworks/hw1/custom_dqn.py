import os
import sys
import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, List


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: object):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        total = self.total()

        if self.n_entries == 0 or total == 0:
            raise ValueError("SumTree is empty — no samples available yet.")

        s = np.clip(s, 0, total - 1e-5)
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        if data_idx >= self.n_entries or self.data[data_idx] is None:
            data_idx = np.random.randint(0, self.n_entries)
            idx = data_idx + self.capacity - 1

        return idx, self.tree[idx], self.data[data_idx]
        

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100_000, gamma: float = 0.99, alpha: float = 0.6, eps: float = 1e-6) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def _get_priority(self, td_error: float) -> float:
        return (abs(td_error) + self.eps) ** self.alpha

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        transition = state, action, reward, next_state, done
        max_prio = float(np.max(self.tree.tree[-self.tree.capacity:]))
        if max_prio == 0:
            max_prio = 1.0
        self.tree.add(max_prio, transition)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs + 1e-6) ** (-beta)
        weights /= (weights.max() + 1e-6)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            self.tree.update(idx, self._get_priority(float(td_error)))

    def __len__(self) -> int:
        return self.tree.n_entries


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.v_stream = nn.Linear(hidden_dim, 1)
        self.a_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        v = self.v_stream(x)
        a = self.a_stream(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
        
        
class CustomDQN:
    def __init__(
        self,
        env,
        learning_rate=1e-3,
        learning_starts=0,
        buffer_size=50000,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        batch_size=32,
        hidden_dim=64,
        gamma=0.99,
        per_alpha=0.6,
        per_beta=0.4,
        per_gamma=0.99,
        per_eps=1e-6,
    ):
        self.env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.gamma = gamma

        self.fraction = 0
        self.epsilon = 1
        self.min_epsilon = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        
        self.per_beta = per_beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.online_net = Network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = Network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        self.replay = PrioritizedReplayBuffer(buffer_size, gamma=per_gamma, alpha=per_alpha, eps=per_eps)

    def _to_tensor(self, batch, dtype=torch.float32):
        return torch.tensor(batch, dtype=dtype, device=self.device)

    def update_exploration(self):
        if self.fraction < self.exploration_fraction:
            fraction = self.fraction / self.exploration_fraction
            self.epsilon = 1 + fraction * (self.min_epsilon - 1) 
        else:
            self.epsilon = self.min_epsilon

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        state = self._to_tensor(np.expand_dims(obs, 0))
        Q_s = self.online_net.forward(state).squeeze()

        self.update_exploration()
        return torch.argmax(Q_s).item()

    def sample_from_replay(self):
        beta = self.per_beta + self.fraction * (1.0 - self.per_beta)
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.batch_size, beta=beta)
        return (self._to_tensor(states), 
                self._to_tensor(actions, dtype=torch.int64), 
                self._to_tensor(rewards), 
                self._to_tensor(next_states),
                self._to_tensor(dones),
                self._to_tensor(weights),
                indices,
               )

    def update_network(self):
        states, actions, rewards, next_states, dones, weights, indices = self.sample_from_replay()

        with torch.no_grad():
            V_sn = self.target_net.forward(next_states).amax(dim=1)
            targets = rewards + (1 - dones) * self.gamma * V_sn

        Q_sa = self.online_net.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = targets - Q_sa
        loss = (weights * td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        td_np = td_errors.abs().detach().cpu().numpy()
        self.replay.update_priorities(np.asarray(indices, np.int64), td_np)

    def learn(self, total_timesteps):
        obs, _ = self.env.reset()
        episode_len = 0

        for step in range(1, total_timesteps + 1):
            self.fraction = step / total_timesteps
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay.push(obs, action, reward, next_obs, float(terminated))
            obs = next_obs

            episode_len += 1

            if len(self.replay) >= self.batch_size and step >= self.learning_starts and step % self.train_freq == 0:
                self.update_network()

            if step % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if terminated or truncated:
                obs, _ = self.env.reset()
                episode_len = 0

