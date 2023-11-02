"""Tests for R2D2 actors."""
import time

import torch.multiprocessing as mp

from minidrl.r2d2.r2d2 import R2D2Config, run_actor
from minidrl.r2d2.replay import run_replay_process
from minidrl.r2d2.run_gym import get_gym_env_creator_fn, gym_model_loader


def run_actor_test(config):
    config.env_creator_fn_getter = get_gym_env_creator_fn
    config.model_loader = gym_model_loader

    model = config.load_model()

    mp_ctxt = mp.get_context("spawn")
    terminate_event = mp_ctxt.Event()
    actor_replay_queue = mp_ctxt.JoinableQueue()
    actor_recv_queue = mp_ctxt.JoinableQueue()
    actor_send_queues = [mp_ctxt.JoinableQueue() for _ in range(config.num_actors)]
    replay_send_queue = mp_ctxt.JoinableQueue()
    replay_recv_queue = mp_ctxt.JoinableQueue()

    # spawn replay process
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_replay_queue,
            replay_send_queue,
            replay_recv_queue,
            terminate_event,
        ),
    )
    replay_process.start()

    actors = []
    for actor_idx in range(config.num_actors):
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                config,
                actor_replay_queue,
                actor_recv_queue,
                actor_send_queues[actor_idx],
                terminate_event,
            ),
        )
        actors.append(actor)
        actor.start()

    # wait for actor to request latest params
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    for _ in range(config.num_actors):
        request = actor_recv_queue.get()
        actor_recv_queue.task_done()
        assert request[0] == "get_latest_params"
        actor_idx = request[1]
        assert isinstance(actor_idx, int), 0 <= actor_idx < config.num_actors
        actor_send_queues[actor_idx].put(("params", model_state))

    # give time for actor to add to replay
    replay_send_queue.put(("get_buffer_size", None))
    buffer_size = replay_recv_queue.get()
    replay_recv_queue.task_done()
    assert buffer_size >= config.learning_starts

    print("main: sending terminate signal")
    terminate_event.set()

    # give time for processes to terminate/stop adding to queues
    time.sleep(2)

    print("main: draining queues")
    for q in [
        actor_replay_queue,
        replay_send_queue,
        replay_recv_queue,
        actor_recv_queue,
    ] + actor_send_queues:
        while not q.empty():
            q.get()
            q.task_done()
        q.close()

    print("main: joining replay process")
    replay_process.join()
    print("main: joining actors")
    for actor in actors:
        actor.join()


def test_run_actor():
    """Tests that actor adds to replay correctly."""
    print("\ntest_run_actor")
    config = R2D2Config(
        env_id="CartPole-v1",
        replay_buffer_size=100,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        batch_size=1,
        learning_starts=16,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        actor_update_interval=int(1e9),  # large to prevent multiple actor updates
    )
    run_actor_test(config)


def test_run_multiple_actors():
    """Tests running multiple actors."""
    print("\ntest_run_multiple_actors")
    config = R2D2Config(
        env_id="CartPole-v1",
        replay_buffer_size=8,
        num_actors=2,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        batch_size=1,
        learning_starts=1,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        actor_update_interval=int(1e9),  # large to prevent multiple actor updates
    )
    run_actor_test(config)


if __name__ == "__main__":
    # run multiple times to try catch race condition concurrency bugs
    for _ in range(3):
        test_run_actor()
    for _ in range(3):
        test_run_multiple_actors()
