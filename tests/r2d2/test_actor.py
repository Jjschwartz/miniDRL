"""Tests for R2D2 actors."""
import time

import torch.multiprocessing as mp
from minidrl.r2d2.r2d2 import run_actor
from minidrl.r2d2.replay import run_replay_process
from minidrl.r2d2.utils import R2D2Config


def test_run_actor():
    """Tests that actor adds to replay correctly."""
    print("\n\ntest_run_actor")
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=8,
        num_actors=1,
        num_envs_per_actor=1,
        seq_len=4,
        burnin_len=2,
        batch_size=1,
        learning_starts=1,
        replay_priority_exponent=0.9,
        replay_priority_noise=1e-3,
        importance_sampling_exponent=0.6,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        actor_update_interval=int(1e9),  # large to prevent multiple actor updates
    )

    model = config.load_model()

    mp_ctxt = mp.get_context("spawn")

    actor_replay_queue = mp_ctxt.Queue()
    actor_recv_queue = mp_ctxt.Queue()
    actor_send_queues = [mp_ctxt.Queue() for _ in range(config.num_actors)]
    replay_send_queue = mp_ctxt.Queue()
    replay_recv_queue = mp_ctxt.Queue()

    # spawn replay process
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_replay_queue,
            replay_send_queue,
            replay_recv_queue,
        ),
    )
    replay_process.start()

    time.sleep(1)

    actor = mp_ctxt.Process(
        target=run_actor,
        args=(0, config, actor_replay_queue, actor_recv_queue, actor_send_queues[0]),
    )
    actor.start()

    # wait for actor to request latest params
    request = actor_recv_queue.get()
    assert request[0] == "get_latest_params"
    assert request[1] == 0
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    actor_send_queues[0].put(("params", model_state))

    # give time for actor to add to replay
    replay_send_queue.put(("get_buffer_size", None))
    buffer_size = replay_recv_queue.get()
    assert buffer_size >= config.learning_starts

    print("main: sending terminate signal to replay process")
    replay_send_queue.put(("terminate", None))
    replay_process.join()

    print("main: sending terminate signal to actor")
    actor_send_queues[0].put(("terminate", None))

    print("main: draining queues")
    while not actor_recv_queue.empty():
        actor_recv_queue.get()
    while not actor_replay_queue.empty():
        actor_replay_queue.get()

    print("main: joining actor")
    actor.join()

    # close queues
    print("main: closing queues")
    actor_replay_queue.close()
    actor_recv_queue.close()
    for q in actor_send_queues:
        q.close()
    replay_send_queue.close()
    replay_recv_queue.close()


def test_run_multiple_actors():
    """Tests running multiple actors."""
    print("\n\ntest_run_multiple_actors")
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
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
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        actor_update_interval=int(1e9),  # large to prevent multiple actor updates
    )

    model = config.load_model()

    mp_ctxt = mp.get_context("spawn")

    actor_replay_queue = mp_ctxt.Queue()
    actor_recv_queue = mp_ctxt.Queue()
    actor_send_queues = [mp_ctxt.Queue() for _ in range(config.num_actors)]
    replay_send_queue = mp_ctxt.Queue()
    replay_recv_queue = mp_ctxt.Queue()

    # spawn replay process
    replay_process = mp_ctxt.Process(
        target=run_replay_process,
        args=(
            config,
            actor_replay_queue,
            replay_send_queue,
            replay_recv_queue,
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
            ),
        )
        actors.append(actor)
        actor.start()

    # wait for actor to request latest params
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    for _ in range(config.num_actors):
        request = actor_recv_queue.get()
        assert request[0] == "get_latest_params"
        actor_idx = request[1]
        assert isinstance(actor_idx, int), 0 <= actor_idx < config.num_actors
        actor_send_queues[actor_idx].put(("params", model_state))

    # give time for actor to add to replay
    replay_send_queue.put(("get_buffer_size", None))
    buffer_size = replay_recv_queue.get()
    assert buffer_size >= config.learning_starts

    print("main: sending terminate signal to replay process")
    replay_send_queue.put(("terminate", None))
    replay_process.join()

    print("main: sending terminate signal to actor")
    for actor_idx in range(config.num_actors):
        actor_send_queues[actor_idx].put(("terminate", None))

    print("main: draining queues")
    while not actor_recv_queue.empty():
        actor_recv_queue.get()
    while not actor_replay_queue.empty():
        actor_replay_queue.get()

    print("main: joining actor")
    for actor in actors:
        actor.join()

    # close queues
    print("main: closing queues")
    actor_replay_queue.close()
    actor_recv_queue.close()
    for q in actor_send_queues:
        q.close()
    replay_send_queue.close()
    replay_recv_queue.close()


if __name__ == "__main__":
    test_run_actor()
    test_run_multiple_actors()
