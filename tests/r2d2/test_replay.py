"""Tests for R2D2 replay buffer."""
import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from minidrl.r2d2.r2d2 import R2D2Config
from minidrl.r2d2.replay import R2D2PrioritizedReplay, SumTree, run_replay_process


# The size of the LSTM in the default gym R2D2 network
LSTM_SIZE = 128


def get_gym_env_creator_fn(
    config: R2D2Config,
    env_idx: int,
    actor_idx: int | None = None,
) -> callable:
    """Get environment creator function."""

    def thunk():
        return gym.make(config.env_id)

    return thunk


def test_sum_tree():
    capacity = 8
    tree = SumTree(capacity)

    assert tree.capacity == capacity
    assert tree.first_leaf_idx == capacity - 1
    assert tree.storage.shape == (capacity * 2 - 1,)
    assert tree.values.shape == (capacity,)
    assert tree.total == 0.0

    tree.set([0], [1.0])
    assert tree.total == 1.0
    assert tree.storage[tree.first_leaf_idx] == 1.0
    assert tree.values[0] == 1.0
    assert tree.storage[1] == 1.0
    assert tree.storage[3] == 1.0

    tree.set([1], [2.0])
    assert tree.total == 3.0
    assert tree.storage[tree.first_leaf_idx] == 1.0
    assert tree.storage[tree.first_leaf_idx + 1] == 2.0
    assert tree.values[0] == 1.0
    assert tree.values[1] == 2.0
    assert tree.storage[1] == 3.0
    assert tree.storage[3] == 3.0

    indices = tree.find([0.0, 0.9, 1.0, 2.0, 2.9, 3.0])
    assert indices == [0, 0, 1, 1, 1, capacity - 1]

    tree.set([2, 3], [1.0, 0.5])
    assert tree.total == 4.5
    assert tree.storage[tree.first_leaf_idx] == 1.0
    assert tree.storage[tree.first_leaf_idx + 1] == 2.0
    assert tree.storage[tree.first_leaf_idx + 2] == 1.0
    assert tree.storage[tree.first_leaf_idx + 3] == 0.5
    assert tree.values[0] == 1.0
    assert tree.values[1] == 2.0
    assert tree.values[2] == 1.0
    assert tree.values[3] == 0.5
    assert tree.storage[1] == 4.5
    assert tree.storage[3] == 3.0
    assert tree.storage[4] == 1.5

    indices = tree.find([0.0, 0.9, 1.0, 2.0, 2.9, 3.0, 3.9, 4.0, 4.4, 4.5])
    assert indices == [0, 0, 1, 1, 1, 2, 2, 3, 3, capacity - 1]

    tree.set([4, 5, 6, 7], [1.0, 0.5, 0.25, 0.125])
    assert tree.total == 6.375

    tree.set([0], [2.0])
    assert tree.total == 7.375


def _check_replay_init(replay, config, obs_shape):
    C = config.replay_size
    T = config.seq_len + config.burnin_len + 1
    assert replay.capacity == C
    assert replay.total_seq_len == T
    assert replay.num_added == 0
    assert replay.obs_storage.shape == (T, C, *obs_shape)
    assert replay.action_storage.shape == (T, C)
    assert replay.reward_storage.shape == (T, C)
    assert replay.done_storage.shape == (T, C)
    assert replay.lstm_h_storage.shape == (1, C, config.lstm_size)
    assert replay.lstm_c_storage.shape == (1, C, config.lstm_size)
    assert replay.size == 0


def _get_random_input(rng, env, T, B, lstm_size):
    obs = env.reset()[0]
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete)

    obs = torch.from_numpy(np.stack([np.stack([obs] * B)] * T))
    actions = torch.from_numpy(rng.randint(0, act_space.n, (T, B), dtype=np.int64))
    rewards = torch.from_numpy(rng.randn(T, B).astype(dtype=np.float32))
    dones = torch.from_numpy(rng.randint(0, 2, (T, B), dtype=np.int8))
    lstm_h = torch.from_numpy(rng.randn(1, B, lstm_size).astype(dtype=np.float32))
    lstm_c = torch.from_numpy(rng.randn(1, B, lstm_size).astype(dtype=np.float32))
    return obs, actions, rewards, dones, lstm_h, lstm_c


def test_r2d2_prioritized_replay_single_sample():
    rng = np.random.RandomState(0)
    torch.manual_seed(0)

    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        replay_size=8,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        batch_size=8,
        learning_starts=8,
        lstm_size_=LSTM_SIZE,
    )
    env = gym.make(env_id)
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)
    obs_shape = obs_space.shape

    replay = R2D2PrioritizedReplay(obs_space, config)

    _check_replay_init(replay, config, obs_shape)

    T = config.seq_len + config.burnin_len + 1
    B = config.num_envs_per_actor

    # Add a single transition
    obs, actions, rewards, dones, lstm_h, lstm_c = _get_random_input(
        rng, env, T, B, config.lstm_size
    )

    priority = np.ones((B,))
    replay.add(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        lstm_h=lstm_h,
        lstm_c=lstm_c,
        priority=priority,
    )
    assert replay.num_added == 1
    assert replay.size == 1
    assert torch.isclose(replay.obs_storage[:, 0], obs).all()
    assert torch.isclose(replay.action_storage[:, 0], actions[:, 0]).all()
    assert torch.isclose(replay.reward_storage[:, 0], rewards[:, 0]).all()
    assert torch.isclose(replay.done_storage[:, 0], dones[:, 0]).all()
    assert torch.isclose(replay.lstm_h_storage[:, 0], lstm_h[:, 0]).all()
    assert torch.isclose(replay.lstm_c_storage[:, 0], lstm_c[:, 0]).all()
    assert replay.sum_tree.total == 1.0

    transitions = replay.get([0], device="cpu")
    assert transitions[0].shape == (T, 1, *obs_shape)
    assert transitions[1].shape == (T, 1)
    assert transitions[2].shape == (T, 1)
    assert transitions[3].shape == (T, 1)
    assert transitions[4].shape == (1, 1, config.lstm_size)
    assert transitions[5].shape == (1, 1, config.lstm_size)
    assert torch.isclose(transitions[0], obs).all()
    assert torch.isclose(transitions[1], actions).all()
    assert torch.isclose(transitions[2], rewards).all()
    assert torch.isclose(transitions[3], dones).all()
    assert torch.isclose(transitions[4], lstm_h).all()
    assert torch.isclose(transitions[5], lstm_c).all()

    samples, indices, weights = replay.sample(1, device="cpu")
    assert samples[0].shape == (T, 1, *obs_shape)
    assert samples[1].shape == (T, 1)
    assert samples[2].shape == (T, 1)
    assert samples[3].shape == (T, 1)
    assert samples[4].shape == (1, 1, config.lstm_size)
    assert samples[5].shape == (1, 1, config.lstm_size)
    assert len(indices) == 1
    assert weights.shape == (1,)


def test_r2d2_prioritized_replay_batch_sample():
    rng = np.random.RandomState(0)
    torch.manual_seed(0)

    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        num_actors=1,
        num_envs_per_actor=4,
        seq_len=4,
        burnin_len=2,
        replay_size=8,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        batch_size=8,
        learning_starts=8,
        lstm_size_=LSTM_SIZE,
    )
    env = gym.make(env_id)
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)
    obs_shape = obs_space.shape

    replay = R2D2PrioritizedReplay(obs_space, config)

    _check_replay_init(replay, config, obs_shape)

    T = config.seq_len + config.burnin_len + 1
    B = config.num_envs_per_actor

    # Add a single transition
    obs, actions, rewards, dones, lstm_h, lstm_c = _get_random_input(
        rng, env, T, B, config.lstm_size
    )
    priority = np.ones((B,))
    replay.add(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        lstm_h=lstm_h,
        lstm_c=lstm_c,
        priority=priority,
    )

    assert replay.num_added == B
    assert replay.size == B
    assert torch.isclose(replay.obs_storage[:, :B], obs).all()
    assert torch.isclose(replay.action_storage[:, :B], actions[:, :B]).all()
    assert torch.isclose(replay.reward_storage[:, :B], rewards[:, :B]).all()
    assert torch.isclose(replay.done_storage[:, :B], dones[:, :B]).all()
    assert torch.isclose(replay.lstm_h_storage[:, :B], lstm_h[:, :B]).all()
    assert torch.isclose(replay.lstm_c_storage[:, :B], lstm_c[:, :B]).all()
    assert replay.sum_tree.total == 1.0 * B

    transitions = replay.get(list(range(B)), device="cpu")
    assert transitions[0].shape == (T, B, *obs_shape)
    assert transitions[1].shape == (T, B)
    assert transitions[2].shape == (T, B)
    assert transitions[3].shape == (T, B)
    assert transitions[4].shape == (1, B, config.lstm_size)
    assert transitions[5].shape == (1, B, config.lstm_size)
    assert torch.isclose(transitions[0], obs).all()
    assert torch.isclose(transitions[1], actions).all()
    assert torch.isclose(transitions[2], rewards).all()
    assert torch.isclose(transitions[3], dones).all()
    assert torch.isclose(transitions[4], lstm_h).all()
    assert torch.isclose(transitions[5], lstm_c).all()

    samples, indices, weights = replay.sample(B, device="cpu")
    assert samples[0].shape == (T, B, *obs_shape)
    assert samples[1].shape == (T, B)
    assert samples[2].shape == (T, B)
    assert samples[3].shape == (T, B)
    assert samples[4].shape == (1, B, config.lstm_size)
    assert samples[5].shape == (1, B, config.lstm_size)
    assert len(indices) == B
    assert weights.shape == (B,)


def test_r2d2_distributed_replay():
    rng = np.random.RandomState(0)
    torch.manual_seed(0)

    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        num_actors=2,
        num_envs_per_actor=4,
        seq_len=4,
        burnin_len=2,
        replay_size=8,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        batch_size=8,
        learning_starts=8,
        lstm_size_=LSTM_SIZE,
    )
    config.env_creator_fn_getter = get_gym_env_creator_fn
    env = gym.make(env_id)
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    mp_ctxt = mp.get_context("spawn")
    terminate_event = mp_ctxt.Event()
    actor_to_replay_queue = mp_ctxt.JoinableQueue(maxsize=config.num_actors * 2)
    learner_to_replay_queue = mp_ctxt.JoinableQueue()
    replay_to_learner_queue = mp_ctxt.JoinableQueue()
    log_queue = mp_ctxt.JoinableQueue()

    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_to_replay_queue,
            learner_to_replay_queue,
            replay_to_learner_queue,
            log_queue,
            terminate_event,
        ),
    )
    replay_process.start()

    T = config.seq_len + config.burnin_len + 1
    B = config.num_envs_per_actor

    for _ in range((config.batch_size // config.num_envs_per_actor) + 1):
        # Add batch of transitions
        obs, actions, rewards, dones, lstm_h, lstm_c = _get_random_input(
            rng, env, T, B, config.lstm_size
        )
        priority = np.ones((B,))
        actor_to_replay_queue.put(
            (
                obs,
                actions,
                rewards,
                dones,
                lstm_h,
                lstm_c,
                priority,
            )
        )

    # Get batch of transitions
    learner_to_replay_queue.put(("sample", B))
    samples, indices, weights = replay_to_learner_queue.get()
    replay_to_learner_queue.task_done()

    assert samples[0].shape == (T, B, *obs_space.shape)
    assert samples[1].shape == (T, B)
    assert samples[2].shape == (T, B)
    assert samples[3].shape == (T, B)
    assert samples[4].shape == (1, B, config.lstm_size)
    assert samples[5].shape == (1, B, config.lstm_size)
    assert len(indices) == B
    assert weights.shape == (B,)

    updated_priorities = np.random.rand(B).tolist()
    learner_to_replay_queue.put(("update_priorities", indices, updated_priorities))

    # delete any shared memory references, so shutdown happens correctly
    del samples, indices, weights

    # Close replay process
    terminate_event.set()

    learner_to_replay_queue.join()
    actor_to_replay_queue.join()

    while not log_queue.empty():
        log_queue.get()
        log_queue.task_done()
    log_queue.close()

    replay_process.join()
    actor_to_replay_queue.close()
    learner_to_replay_queue.close()
    replay_to_learner_queue.close()


if __name__ == "__main__":
    test_sum_tree()
    test_r2d2_prioritized_replay_single_sample()
    test_r2d2_prioritized_replay_batch_sample()
    test_r2d2_distributed_replay()
