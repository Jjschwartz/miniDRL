"""Tests for R2D2 actors."""
import time

import gymnasium as gym
import torch.multiprocessing as mp

from d2rl.r2d2.r2d2 import run_actor
from d2rl.r2d2.replay import R2D2PrioritizedReplay
from d2rl.r2d2.utils import R2D2Config


def test_run_actor():
    """Tests that we can run actor without multiprocessing."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        batch_size=8,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=14,
    )
    env = config.make_env()
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    learner_model = config.load_model()
    input_queue = mp.Queue()
    replay = R2D2PrioritizedReplay(obs_space, config)
    actor_storage, actor_lock = replay.get_actor_storage(0)

    run_actor(
        actor_idx=0,
        config=config,
        learner_model=learner_model,
        input_queue=input_queue,
        actor_storage=actor_storage,
        actor_lock=actor_lock,
    )
    input_queue.close()


def test_run_and_terminate_actor():
    """Tests that we can run actor on seperate process and terminate it correctly."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=1000,  # make large so we terminate before transition added
        burnin_len=200,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=None,
    )
    env = config.make_env()
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    learner_model = config.load_model()
    replay = R2D2PrioritizedReplay(obs_space, config)
    actor_storage, actor_lock = replay.get_actor_storage(0)

    mp_ctxt = mp.get_context("spawn")
    input_queue = mp_ctxt.Queue()
    actor = mp_ctxt.Process(
        target=run_actor,
        args=(0, config, learner_model, input_queue, actor_storage, actor_lock),
    )
    actor.start()

    # Send a terminate message (can be anything)
    input_queue.put("terminate")
    actor.join()
    input_queue.close()


def test_run_actor_add_to_replay():
    """Tests that actor adds to replay correctly."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=6,  # == sequence length
    )
    env = config.make_env()
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    learner_model = config.load_model()
    input_queue = mp.Queue()
    replay = R2D2PrioritizedReplay(obs_space, config)
    actor_storage, actor_lock = replay.get_actor_storage(0)

    assert replay.size == 0
    run_actor(
        actor_idx=0,
        config=config,
        learner_model=learner_model,
        input_queue=input_queue,
        actor_storage=actor_storage,
        actor_lock=actor_lock,
    )
    assert replay.size == 1

    input_queue.close()


def test_mp_run_actor_add_to_replay():
    """Tests that actor adds to replay correctly."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=6,  # == sequence length
    )
    env = config.make_env()
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    learner_model = config.load_model()
    replay = R2D2PrioritizedReplay(obs_space, config)
    actor_storage, actor_lock = replay.get_actor_storage(0)

    assert replay.size == 0
    mp_ctxt = mp.get_context("spawn")
    input_queue = mp_ctxt.Queue()
    actor = mp_ctxt.Process(
        target=run_actor,
        args=(0, config, learner_model, input_queue, actor_storage, actor_lock),
    )
    actor.start()

    # give time for actor to add to replay
    time.sleep(2)

    assert replay.size == 1

    # wait for actor to execute debug steps
    actor.join()
    input_queue.close()


def test_mp_run_multiple_actors_add_to_replay():
    """Tests that actor adds to replay correctly."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=2,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=6,  # == sequence length
    )
    env = config.make_env()
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Box)

    learner_model = config.load_model()
    replay = R2D2PrioritizedReplay(obs_space, config)

    assert replay.size == 0
    mp_ctxt = mp.get_context("spawn")
    input_queues = []
    actors = []
    for actor_idx in range(config.num_actors):
        input_queue = mp_ctxt.Queue()
        actor_storage, actor_lock = replay.get_actor_storage(actor_idx)
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                config,
                learner_model,
                input_queue,
                actor_storage,
                actor_lock,
            ),
        )
        actors.append(actor)
        input_queues.append(input_queue)
        actor.start()

    # give time for actor to add to replay
    time.sleep(2)

    assert replay.size == 2

    for actor_idx in range(config.num_actors):
        actors[actor_idx].join()
        input_queues[actor_idx].close()


if __name__ == "__main__":
    test_run_actor()
    # test_run_and_terminate_actor()
    # test_run_actor_add_to_replay()
    # test_mp_run_actor_add_to_replay()
    # test_mp_run_multiple_actors_add_to_replay()
