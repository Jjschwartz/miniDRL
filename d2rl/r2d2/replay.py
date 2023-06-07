"""Replay Buffer for R2D2."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from gymnasium import spaces

    from d2rl.r2d2.utils import R2D2Config


class SumTree:
    """A binary tree where non-leaf nodes ar the sum of their children.

    Leaf nodes contain non-negative floats and are set externally. Non-leaf nodes
    are the sum of their children. This data structure allows O(log n) updates and
    O(log n) queries of which index corresponds to a given sum. The main use
    case is sampling from a multinomial distribution with many probabilities
    which are updated a few at a time.

    Reference:
    https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/replay.py#L167
    """

    def __init__(self, capacity: int, storage: torch.Tensor | None = None):
        """Initialize."""
        # For capacity=n, the tree has 2n-1 nodes.
        # The first n-1 nodes are non-leaf nodes, with index 0 corresponding to the
        # root node. The children of non-leaf node i are nodes 2i+1 and 2i+2.
        # The last n nodes are leaf nodes, that contain values.
        self.capacity = capacity
        self.first_leaf_idx = capacity - 1

        if storage is None:
            storage = torch.zeros((2 * capacity - 1,), dtype=torch.float32)
            storage.share_memory_()
        else:
            assert storage.shape == (2 * capacity - 1,)
        self.storage = storage
        self.values = self.storage[self.first_leaf_idx :]

    @property
    def total(self) -> float:
        """Total value of the tree."""
        return self.storage[0]

    def get(self, indices: List[int] | torch.Tensor) -> torch.Tensor:
        """Get values from the tree."""
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        return self.values[indices]

    def set(
        self, indices: List[int] | torch.Tensor, values: List[float] | torch.Tensor
    ):
        """Set values in the tree."""
        if isinstance(indices, (list, np.ndarray)):
            indices = torch.tensor(indices)
        if isinstance(values, (list, np.ndarray)):
            values = torch.tensor(values, dtype=torch.float32)
        self.values[indices] = values
        for i in indices + self.capacity:
            parent = (i - 2) // 2
            while parent >= 0:
                self.storage[parent] = (
                    self.storage[2 * parent + 1] + self.storage[2 * parent + 2]
                )
                parent = (parent - 1) // 2

    def find(self, targets: List[float] | np.ndarray | torch.tensor) -> List[int]:
        """Finds smallest indices where `target <` cumulative value sum up to index.

        If `target >=` the total value of the tree, the index returned will be the
        final leaf node.

        Arguments
        ---------
        targets : The target values to search for.

        Returns
        -------
        indices : For each target, the smallest index such that target is strictly less
            than the cumulative sum of values up to and including that index.

        """
        indices = []
        for target in targets:
            # binary search starting from the root node
            idx = 0
            while idx < self.first_leaf_idx:
                left = 2 * idx + 1
                if target < self.storage[left]:
                    idx = left
                else:
                    target -= self.storage[left]
                    idx = left + 1
            indices.append(idx - self.first_leaf_idx)
        return indices


class R2D2PrioritizedReplay:
    """Prioritized Experience Replay.

    Uses proportional prioritization as described in
    "Priotizied Experience Replay" https://arxiv.org/pdf/1511.05952.pdf

    Stores fixed-length sequences of transitions (o, a, r, done). These sequences
    consist of a burn-in number of steps plus the actual training sequence, as per the
    R2D2 paper. Once thing we do differently here is that we allow sequences to cross
    episode boundaries. This means all sequences are the same length, and we can
    easily handle running vectorized environments within each actor.

    With each sequence the state of the LSTM at the start of the sequence is also
    stored. Note, for zero-state initialization this LSTM state will be zeros.

    Each sequence is assigned a priority, which determines the probability that the
    sequence will be sampled.

    To allow for multiple actors adding transitions to the buffer in parallel, the
    storage is partitioned into num_actors parts, with each actor only adding to
    their own partition. This means that each actor can add transitions without
    needing to synchronize with the other actors.

    The storage uses shared memory tensors to allow for efficient access by multiple
    parallel processes.

    Each entry of a sequence contains the following:

    - observation: o_t
    - prev action: a_tm1
    - prev reward: r_tm1
    - prev step done: done_tm1

    Since R2D2 needs a_t, r_t, and done_t, for computing loss and priorities for time
    `t`, we store sequences of length T+1, where T is the burin +sequence length.

    """

    def __init__(self, obs_space: spaces.Box, config: R2D2Config):
        """Initialize."""
        self.config = config
        # capacity = number of sequences
        self.capacity = config.replay_buffer_size
        self.actor_capacity = self.capacity // config.num_actors
        self.total_seq_len = config.seq_len + config.burnin_len + 1

        C, T = self.capacity, self.total_seq_len
        self.obs_storage = torch.zeros(
            (T, C, *obs_space.shape),
            dtype=torch.from_numpy(obs_space.sample()).dtype,
        )
        self.action_storage = torch.zeros((T, C), dtype=torch.int8)
        self.reward_storage = torch.zeros((T, C), dtype=torch.float32)
        self.done_storage = torch.zeros((T, C), dtype=torch.bool)
        self.lstm_h_storage = torch.zeros((1, C, config.lstm_size), dtype=torch.float32)
        self.lstm_c_storage = torch.zeros((1, C, config.lstm_size), dtype=torch.float32)
        self.num_added = torch.zeros((config.num_actors,), dtype=torch.long)

        self.obs_storage.share_memory_()
        self.action_storage.share_memory_()
        self.reward_storage.share_memory_()
        self.done_storage.share_memory_()
        self.lstm_h_storage.share_memory_()
        self.lstm_c_storage.share_memory_()
        self.num_added.share_memory_()

        self.sum_trees = [
            SumTree(self.actor_capacity) for _ in range(config.num_actors)
        ]
        # Locks for preventing multiple updates to the same tree at the same time.
        self.locks = [mp.Lock() for _ in range(config.num_actors)]

    def actor_start_idx(self, actor_idx: int) -> int:
        """Get the start index for the storage assigned to an actor."""
        assert 0 <= actor_idx < self.config.num_actors
        return actor_idx * self.actor_capacity

    def get(self, indices: List[int], device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Get transitions from the replay buffer.

        T = seq len of the transition (should be equal to self.total_seq_len)
        L = num layers in the LSTM (typically 1)

        Arguments
        ---------
        indices : Indices into storage for the transitions to get.
        device : The device to put the tensors on.

        Returns
        -------
        obs : The observations. Shape=(T+1, len(indices), *obs_space.shape)
        actions : The actions. Shape=(T+1, len(indices))
        rewards : The rewards. Shape=(T+1, len(indices))
        done : The done flags. Shape=(T+1, len(indices))
        lstm_h : The LSTM hidden state. Shape=(L, len(indices), lstm_size)
        lstm_c : The LSTM cell state. Shape=(L, len(indices), lstm_size)
        """
        return (
            self.obs_storage[:, indices].to(device),
            self.action_storage[:, indices].to(device),
            self.reward_storage[:, indices].to(device),
            self.done_storage[:, indices].to(device),
            self.lstm_h_storage[:, indices].to(device),
            self.lstm_c_storage[:, indices].to(device),
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tuple[torch.Tensor, ...], List[int], torch.Tensor]:
        """Sample a batch of transitions from the replay buffer.

        Arguments
        ---------
        batch_size : Number of transitions to sample.
        device : Device to put the tensors on.

        Returns
        -------
        samples : The sampled batch of transitions.
        indices : Indices of the sampled transitions in the replay buffer.
        weights : Importance sampling weights for the batch.
        """
        assert batch_size > 0
        assert self.size >= batch_size

        # sample indices according to priorities
        indices = []
        priorities = []
        for actor_idx in range(self.config.num_actors):
            # sample subset of batch from each actor's sum tree
            sum_tree = self.sum_trees[actor_idx]
            targets = np.random.uniform(
                0, sum_tree.total, size=batch_size // self.config.num_actors
            )
            with self.locks[actor_idx]:
                actor_indices = sum_tree.find(targets)
                actor_priorities = sum_tree.get(actor_indices)
            priorities.append(actor_priorities)
            indices.extend([i + actor_idx * self.actor_capacity for i in actor_indices])
        priorities = torch.concatenate(priorities)

        # calculate priority proportional probabilities
        total_priority = sum([sum_tree.total for sum_tree in self.sum_trees])
        uniform_prob = 1.0 / self.size
        noise = self.config.replay_priority_noise
        prioritized_probs = (
            noise * uniform_prob + (1.0 - noise) * priorities / total_priority
        )

        # calculate importance sampling weights to correct for bias introduced by
        # prioritized sampling
        weights = (
            uniform_prob / prioritized_probs
        ) ** self.config.importance_sampling_exponent
        # normalize weights so that the maximum weight is 1 (for stability)
        weights /= weights.max()
        if not torch.isfinite(weights).all():
            raise ValueError("Weights are not finite: %s." % weights)

        samples = self.get(indices, device=device)
        return samples, indices, weights.to(device)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update the priorities of transitions in the replay buffer."""
        assert len(indices) == len(priorities)

        # sort indices and priorities by index so they can be partitioned by actor
        ordered_values = list(zip(indices, priorities))
        ordered_values.sort(key=lambda x: x[0])
        priorities = torch.tensor([p for _, p in ordered_values], dtype=torch.float32)
        indices = torch.tensor([i for i, _ in ordered_values], dtype=torch.long)

        # By default 0 ** 0 is 1 but we never want indices with priority zero to be
        # sampled, even if the priority exponent is zero.
        priorities = torch.where(
            priorities == 0.0, 0.0, priorities**self.config.replay_priority_exponent
        )

        # Partition indices and priorities by actor storage.
        actor_partitions = []
        start = 0
        for actor_idx in range(self.config.num_actors):
            start_idx = self.actor_start_idx(actor_idx)
            end_idx = start_idx + self.actor_capacity
            end = start
            while end < len(indices) and indices[end] < end_idx:
                end += 1
            actor_partitions.append((start, end))
            start = end

        # Update priorities in the sum tree for each actor.
        for actor_idx in range(self.config.num_actors):
            start, end = actor_partitions[actor_idx]
            with self.locks[actor_idx]:
                self.sum_trees[actor_idx].set(indices[start:end], priorities[start:end])

    @property
    def size(self) -> int:
        """Return the number of transitions in the replay buffer."""
        return sum(
            min(self.num_added[i], self.actor_capacity)
            for i in range(self.config.num_actors)
        )

    def get_actor_storage(
        self, actor_idx: int
    ) -> Tuple[Tuple[torch.Tensor, ...], mp.Lock]:
        """Get the storage tensors for a specific actor."""
        start_idx = self.actor_start_idx(actor_idx)
        end_idx = start_idx + self.actor_capacity
        return {
            "obs_storage": self.obs_storage[:, start_idx:end_idx],
            "action_storage": self.action_storage[:, start_idx:end_idx],
            "reward_storage": self.reward_storage[:, start_idx:end_idx],
            "done_storage": self.done_storage[:, start_idx:end_idx],
            "lstm_h_storage": self.lstm_h_storage[:, start_idx:end_idx],
            "lstm_c_storage": self.lstm_c_storage[:, start_idx:end_idx],
            "sum_tree_storage": self.sum_trees[actor_idx].storage,
            "num_added": self.num_added,
        }, self.locks[actor_idx]


class R2D2ActorReplayBuffer:
    """Replay buffer for R2D2 actors.

    This class provides an interface to allow an R2D2 actor to add transitions to the
    shared replay buffer.

    It also maintains it's own sum tree for storing priorities. The sum tree is shared
    with the main replay buffer so that it can be used to sample transitions.

    """

    def __init__(
        self,
        actor_idx: int,
        config: R2D2Config,
        obs_storage: torch.Tensor,
        action_storage: torch.Tensor,
        reward_storage: torch.Tensor,
        done_storage: torch.Tensor,
        lstm_h_storage: torch.Tensor,
        lstm_c_storage: torch.Tensor,
        sum_tree_storage: torch.Tensor,
        num_added: torch.Tensor,
        actor_lock: mp.Lock,
    ):
        self.actor_idx = actor_idx
        self.config = config
        self.capacity = config.replay_buffer_size // config.num_actors
        self.total_seq_len = config.seq_len + config.burnin_len + 1

        self.obs_storage = obs_storage
        self.action_storage = action_storage
        self.reward_storage = reward_storage
        self.done_storage = done_storage
        self.lstm_h_storage = lstm_h_storage
        self.lstm_c_storage = lstm_c_storage
        self.num_added = num_added

        self.sum_tree = SumTree(self.capacity, sum_tree_storage)
        self.lock = actor_lock

    def add(
        self,
        obs: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        dones: torch.tensor,
        lstm_h: torch.tensor,
        lstm_c: torch.tensor,
        priority: float | np.ndarray | torch.tensor,
    ):
        """Add a transition to the replay buffer.

        T = seq len of the transition (should be equal to self.total_seq_len)
        B = batch size (generally equal to self.config.num_actors)
        L = num layers in the LSTM (typically 1)

        Arguments
        ---------
        obs : The observations. Shape=(T+1, B, *obs_space.shape)
        actions : The actions. Shape=(T+1, B)
        rewards : The rewards. Shape=(T+1, B)
        dones : The done flags. Shape=(T+1, B)
        lstm_h : The LSTM hidden state. Shape=(L, B, lstm_size)
        lstm_c : The LSTM cell state. Shape=(L, B, lstm_size)
        priority : The priority of the transition. Shape=(B,)

        """
        assert len(obs) == self.total_seq_len

        batch_size = obs.shape[1]
        index = self.num_added[self.actor_idx] % self.capacity

        if index + batch_size > self.capacity:
            # Need to wrap around to the start of the actor's storage
            end_idx = self.capacity
            rem_idx = end_idx - index
            self.obs_storage[:, index:end_idx] = obs[:, :rem_idx]
            self.action_storage[:, index:end_idx] = actions[:, :rem_idx]
            self.reward_storage[:, index:end_idx] = rewards[:, :rem_idx]
            self.done_storage[:, index:end_idx] = dones[:, :rem_idx]
            self.lstm_h_storage[:, index:end_idx] = lstm_h[:, :rem_idx]
            self.lstm_c_storage[:, index:end_idx] = lstm_c[:, :rem_idx]

            self.obs_storage[:, :rem_idx] = obs[:, rem_idx:]
            self.action_storage[:, :rem_idx] = actions[:, rem_idx:]
            self.reward_storage[:, :rem_idx] = rewards[:, rem_idx:]
            self.done_storage[:, :rem_idx] = dones[:, rem_idx:]
            self.lstm_h_storage[:, :rem_idx] = lstm_h[:, rem_idx:]
            self.lstm_c_storage[:, :rem_idx] = lstm_c[:, rem_idx:]

            indices = list(range(index, end_idx)) + list(range(0, batch_size - rem_idx))
        else:
            self.obs_storage[:, index : index + batch_size] = obs[:, :]
            self.action_storage[:, index : index + batch_size] = actions[:, :]
            self.reward_storage[:, index : index + batch_size] = rewards[:, :]
            self.done_storage[:, index : index + batch_size] = dones[:, :]
            self.lstm_h_storage[:, index : index + batch_size] = lstm_h[:, :]
            self.lstm_c_storage[:, index : index + batch_size] = lstm_c[:, :]
            indices = list(range(index, index + batch_size))

        self.num_added[self.actor_idx] += batch_size

        priority = [priority] if isinstance(priority, float) else list(priority)
        self.update_priorities(indices, priority)

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update the priorities of transitions in the replay buffer."""
        # By default 0 ** 0 is 1 but we never want indices with priority zero to be
        # sampled, even if the priority exponent is zero.
        priorities = np.asarray(priorities)
        priorities = np.where(
            priorities == 0.0, 0.0, priorities**self.config.replay_priority_exponent
        )
        # with self.lock:
        self.sum_tree.set(indices, priorities)

    @property
    def size(self) -> int:
        """Return the number of transitions in the replay buffer."""
        return min(self.num_added[self.actor_idx], self.capacity)
