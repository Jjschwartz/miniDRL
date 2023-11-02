"""Prioritized Replay Buffer for R2D2."""
import time
from multiprocessing.queues import Empty
from typing import Union

import numpy as np
import torch
import torch.multiprocessing as mp
from gymnasium import spaces


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

    def __init__(self, capacity: int):
        # For capacity=n, the tree has 2n-1 nodes.
        # The first n-1 nodes are non-leaf nodes, with index 0 corresponding to the
        # root node. The children of non-leaf node i are nodes 2i+1 and 2i+2.
        # The last n nodes are leaf nodes, that contain values.
        self.capacity = capacity
        self.first_leaf_idx = capacity - 1
        self.storage = torch.zeros((2 * capacity - 1,), dtype=torch.float32)
        self.values = self.storage[self.first_leaf_idx :]

    @property
    def total(self) -> float:
        """Total value of the tree."""
        return self.storage[0]

    def get(self, indices: list[int] | torch.Tensor) -> torch.Tensor:
        """Get values from the tree."""
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        return self.values[indices]

    def set(
        self, indices: list[int] | torch.Tensor, values: list[float] | torch.Tensor
    ):
        """Set values in the tree."""
        if isinstance(indices, (list)):
            indices = torch.tensor(indices)
        if isinstance(values, (list)):
            values = torch.tensor(values, dtype=torch.float32)
        self.values[indices] = values
        for i in indices + self.capacity:
            parent = (i - 2) // 2
            while parent >= 0:
                self.storage[parent] = (
                    self.storage[2 * parent + 1] + self.storage[2 * parent + 2]
                )
                parent = (parent - 1) // 2

    def find(self, targets: Union[list[float], np.ndarray, torch.tensor]) -> list[int]:
        """Finds smallest indices where `target <` cumulative value sum up to index.

        If `target >=` the total value of the tree, the index returned will be the
        final leaf node.

        Arguments
        ---------
        targets
            The target values to search for.

        Returns
        -------
        indices
            For each target, the smallest index such that target is strictly less than
            the cumulative sum of values up to and including that index.
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
    R2D2 paper. One thing we do differently here is that we allow sequences to cross
    episode boundaries. This means all sequences are the same length, and we can
    easily handle running vectorized environments within each actor.

    With each sequence the state of the LSTM at the start of the sequence is also
    stored. Note, for zero-state initialization this LSTM state will just be all zeros.

    Each sequence is assigned a priority, which determines the probability that the
    sequence will be sampled.

    Each entry of a sequence contains the following:

    - observation: o_t
    - prev action: a_tm1
    - prev reward: r_tm1
    - prev step done: done_tm1

    Since R2D2 needs a_t, r_t, and done_t, for computing loss and priorities for time
    `t`, we store sequences of length T+1, where T is the burn-in + sequence length.

    Actors send transitions to the replay buffer via a shared queue.
    """

    def __init__(self, obs_space: spaces.Box, config):
        """Initialize."""
        self.config = config
        # capacity = number of sequences
        self.capacity = config.replay_buffer_size
        self.total_seq_len = config.seq_len + config.burnin_len + 1
        self.num_added = 0

        C, T = self.capacity, self.total_seq_len
        self.obs_storage = torch.zeros(
            (T, C, *obs_space.shape),
            dtype=torch.from_numpy(obs_space.sample()).dtype,
        )
        self.action_storage = torch.zeros((T, C), dtype=torch.long)
        self.reward_storage = torch.zeros((T, C), dtype=torch.float32)
        self.done_storage = torch.zeros((T, C), dtype=torch.int8)
        self.lstm_h_storage = torch.zeros((1, C, config.lstm_size), dtype=torch.float32)
        self.lstm_c_storage = torch.zeros((1, C, config.lstm_size), dtype=torch.float32)

        self.sum_tree = SumTree(self.capacity)

    def add(
        self,
        obs: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        dones: torch.tensor,
        lstm_h: torch.tensor,
        lstm_c: torch.tensor,
        priority: Union[float, np.ndarray, torch.tensor],
    ):
        """Add a transition to the replay buffer.

        T = seq len of the transition (should be equal to self.total_seq_len)
        B = batch size (generally equal to num_envs_per_actor)
        L = num layers in the LSTM (typically 1)

        Arguments
        ---------
        obs
            The observations. Shape=(T+1, B, *obs_space.shape)
        actions
            The actions. Shape=(T+1, B)
        rewards
            The rewards. Shape=(T+1, B)
        dones
            The done flags. Shape=(T+1, B)
        lstm_h
            The LSTM hidden state. Shape=(L, B, lstm_size)
        lstm_c
            The LSTM cell state. Shape=(L, B, lstm_size)
        priority
            The priority of the transition. Shape=(B,)
        """
        assert obs.shape[0] == self.total_seq_len

        batch_size = obs.shape[1]
        index = self.num_added % self.capacity

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

            self.obs_storage[:, : batch_size - rem_idx] = obs[:, rem_idx:]
            self.action_storage[:, : batch_size - rem_idx] = actions[:, rem_idx:]
            self.reward_storage[:, : batch_size - rem_idx] = rewards[:, rem_idx:]
            self.done_storage[:, : batch_size - rem_idx] = dones[:, rem_idx:]
            self.lstm_h_storage[:, : batch_size - rem_idx] = lstm_h[:, rem_idx:]
            self.lstm_c_storage[:, : batch_size - rem_idx] = lstm_c[:, rem_idx:]

            indices = list(range(index, end_idx)) + list(range(0, batch_size - rem_idx))
        else:
            self.obs_storage[:, index : index + batch_size] = obs[:, :]
            self.action_storage[:, index : index + batch_size] = actions[:, :]
            self.reward_storage[:, index : index + batch_size] = rewards[:, :]
            self.done_storage[:, index : index + batch_size] = dones[:, :]
            self.lstm_h_storage[:, index : index + batch_size] = lstm_h[:, :]
            self.lstm_c_storage[:, index : index + batch_size] = lstm_c[:, :]
            indices = list(range(index, index + batch_size))

        self.num_added += batch_size

        priority = [priority] if isinstance(priority, float) else list(priority)
        self.update_priorities(indices, priority)

    def get(self, indices: list[int], device: torch.device) -> tuple[torch.Tensor, ...]:
        """Get transitions from the replay buffer.

        T = seq len of the transition (should be equal to self.total_seq_len)
        L = num layers in the LSTM (typically 1)

        Arguments
        ---------
        indices
            Indices into storage for the transitions to get.
        device
            The device to put the tensors on.

        Returns
        -------
        obs
            The observations. Shape=(T+1, len(indices), *obs_space.shape)
        actions
            The actions. Shape=(T+1, len(indices))
        rewards
            The rewards. Shape=(T+1, len(indices))
        done
            The done flags. Shape=(T+1, len(indices))
        lstm_h
            The LSTM hidden state. Shape=(L, len(indices), lstm_size)
        lstm_c
            The LSTM cell state. Shape=(L, len(indices), lstm_size)
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
    ) -> tuple[tuple[torch.Tensor, ...], list[int], torch.Tensor]:
        """Sample a batch of transitions from the replay buffer.

        Arguments
        ---------
        batch_size
            Number of transitions to sample.
        device
            Device to put the sampled tensors on.

        Returns
        -------
        samples
            The sampled batch of transitions.
        indices
            Indices of the sampled transitions in the replay buffer.
        weights
            Importance sampling weights for the batch.
        """
        assert batch_size > 0
        assert self.size >= batch_size

        # sample indices according to priorities
        targets = np.random.uniform(0, self.sum_tree.total, size=batch_size)
        indices = self.sum_tree.find(targets)
        priorities = self.sum_tree.get(indices)

        # calculate priority proportional probabilities
        uniform_prob = 1.0 / self.size
        noise = self.config.replay_priority_noise
        prioritized_probs = (
            noise * uniform_prob + (1.0 - noise) * priorities / self.sum_tree.total
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

    def update_priorities(self, indices: list[int], priorities: list[float]):
        """Update the priorities of transitions in the replay buffer.

        Arguments
        ---------
        indices
            The indices of the transitions to update.
        priorities
            The new priorities of the transitions.
        """
        assert len(indices) == len(priorities)

        # By default 0 ** 0 is 1 but we never want indices with priority zero to be
        # sampled, even if the priority exponent is zero.
        priorities = torch.tensor(priorities, dtype=torch.float32)
        priorities = torch.where(
            priorities == 0.0, 0.0, priorities**self.config.replay_priority_exponent
        )

        self.sum_tree.set(indices, priorities)

    @property
    def size(self) -> int:
        """Return the number of transitions in the replay buffer."""
        return min(self.num_added, self.capacity)


def run_replay_process(
    config,
    actor_queue: mp.JoinableQueue,
    learner_recv_queue: mp.JoinableQueue,
    learner_send_queue: mp.JoinableQueue,
    terminate_event: mp.Event,
):
    """Run the replay process.

    Arguments
    ---------
    config
        R2D2 configuration.
    actor_queue
        Queue for receiving transitions from actors.
    learner_recieve_queue
        Queue for receiving requests from the learner.
    learner_send_queue
        Queue for sending sampled transitions to the learner.
    terminate_event
        Event for signaling terminating of the run.
    """
    print("replay: starting.")

    # Limit replay to using a single CPU thread.
    # This prevents replay from using all available cores, which can lead to contention
    # with actor and learner processes if not enough cores available.
    torch.set_num_threads(1)

    env = config.env_creator_fn_getter(config, 0, 0)()
    obs_space = env.observation_space
    env.close()

    replay_buffer = R2D2PrioritizedReplay(obs_space, config)

    print("replay: waiting for buffer to fill")
    while replay_buffer.size < config.learning_starts and not terminate_event.is_set():
        try:  # noqa: SIM105
            replay_buffer.add(*actor_queue.get(timeout=1))
            actor_queue.task_done()
        except Empty:
            pass

    print(f"replay: buffer full enough (size={replay_buffer.size}), starting training")
    last_report_time = time.time()
    training_start_time = time.time()
    while not terminate_event.is_set():
        # Check for learner request, here we prioritize learner requests over actor
        if not learner_recv_queue.empty():
            request = learner_recv_queue.get()
            assert isinstance(request, tuple), (
                "Bad learner request to replay. Requests must be tuples, with the "
                f"first element being a string. Got: {request}"
            )
            if request[0] == "sample":
                learner_recv_queue.task_done()
                # sample a batch of transitions and send to learner
                batch_size = request[1]
                samples, indices, weights = replay_buffer.sample(
                    batch_size, config.device
                )
                learner_send_queue.put((samples, indices, weights))
            elif request[0] == "update_priorities":
                replay_buffer.update_priorities(*request[1:])
                # free references to shared memory resources
                del request
                learner_recv_queue.task_done()
            elif request[0] == "get_buffer_size":
                learner_recv_queue.task_done()
                learner_send_queue.put(replay_buffer.size)
            elif request[0] == "get_replay_stats":
                learner_recv_queue.task_done()
                seq_per_sec = replay_buffer.num_added / (
                    time.time() - training_start_time
                )
                learner_send_queue.put(
                    {
                        "size": replay_buffer.size,
                        "seqs_added": replay_buffer.num_added,
                        "steps_added": replay_buffer.num_added * config.seq_len,
                        "seq_per_sec": seq_per_sec,
                        "q_size": actor_queue.qsize(),
                    }
                )
            else:
                learner_recv_queue.task_done()
                raise ValueError(f"Unknown learner request: {request}")

        # Receive transition from actors
        if not actor_queue.empty():
            replay_buffer.add(*actor_queue.get())
            actor_queue.task_done()

        if time.time() - last_report_time > 60:
            output = [
                f"\nreplay - size: {replay_buffer.size}/{replay_buffer.capacity}",
                f"total seqs added: {replay_buffer.num_added}",
                f"total steps added: {replay_buffer.num_added * config.seq_len}",
                f"qsize: {actor_queue.qsize()}/{actor_queue._maxsize}",
            ]
            print("\n  ".join(output))
            last_report_time = time.time()

    print("replay - terminate signal recieved.")
    print("replay - waiting for shared resources to be released.")
    learner_send_queue.join()

    print("Replay - exiting.")
