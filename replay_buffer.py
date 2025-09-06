# replay_buffer.py
import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_frames: int,
        epsilon: float,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame_idx = 0

        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity <<= 1
        self.tree = np.zeros(2 * self.tree_capacity - 1, dtype=np.float64)

        self.data: List[Tuple] = [None] * capacity
        self.idx = 0
        self.size = 0
        self.max_priority = 1.0

        logger.info(f"Initialized PER buffer with capacity={capacity}, alpha={alpha}")

    def _beta(self) -> float:
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        return beta

    def _propagate(self, tree_idx: int, change: float) -> None:
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _update_tree(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        data_idx = self.idx
        self.data[data_idx] = (state, action, reward, next_state, done)

        tree_idx = data_idx + self.tree_capacity - 1
        self._update_tree(tree_idx, self.max_priority**self.alpha)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.frame_idx += 1

    def _retrieve(self, idx: int, sum_priorities: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if sum_priorities <= self.tree[left]:
            return self._retrieve(left, sum_priorities)
        else:
            return self._retrieve(right, sum_priorities - self.tree[left])

    def sample(self, batch_size: int,  device: Optional[torch.device] = None) -> Tuple["np.ndarray | torch.Tensor", ...]:
        """Sample a batch of experiences.

        If a ``device`` is provided, the returned arrays are converted to
        ``torch.Tensor`` objects located on that device, which saves repeated
        host->device transfers in the training loop.
        """
        assert self.size >= batch_size, "Not enough samples in buffer"

        total_p = self.tree[0]
        segment = total_p / batch_size

        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices, weights = [], []

        beta = self._beta()
        min_prob = np.min(self.tree[self.tree_capacity - 1 : self.tree_capacity - 1 + self.size]) / total_p
        max_weight = (min_prob * self.size) ** (-beta)

        for idx_batch in range(batch_size):
            left_bound_of_segment = segment * idx_batch
            right_bound_of_segment = segment * (idx_batch + 1)
            cumulative_priority = random.uniform(left_bound_of_segment, right_bound_of_segment)

            node_idx = self._retrieve(0, cumulative_priority)
            data_idx = node_idx - (self.tree_capacity - 1)

            state, action, reward, nxt, done = self.data[data_idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(nxt)
            dones.append(done)

            p_sample = self.tree[node_idx] / total_p
            w = (p_sample * self.size) ** (-beta)
            weights.append(w / max_weight)
            indices.append(node_idx)

        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)
        dones_arr = np.array(dones, dtype=bool)
        indices_arr = np.array(indices, dtype=np.int64)
        weights_arr = np.array(weights, dtype=np.float32)

        if device is not None:
            states_arr = torch.from_numpy(states_arr).to(device)
            actions_arr = torch.from_numpy(actions_arr).to(device)
            rewards_arr = torch.from_numpy(rewards_arr).to(device)
            next_states_arr = torch.from_numpy(next_states_arr).to(device)
            dones_arr = torch.from_numpy(dones_arr).to(device)
            weights_arr = torch.from_numpy(weights_arr).to(device)

        return (
            states_arr,
            actions_arr,
            rewards_arr,
            next_states_arr,
            dones_arr,
            indices_arr,
            weights_arr,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, error in zip(indices, td_errors):
            new_p = (abs(error) + self.epsilon) ** self.alpha
            self._update_tree(int(idx), new_p)

            self.max_priority = max(self.max_priority, new_p)

    def __len__(self) -> int:
        return self.size
