"""Tests for R2D2 actors."""
import time

import torch.multiprocessing as mp
from minidrl.r2d2.r2d2 import R2D2Config, run_actor
from minidrl.r2d2.replay import run_replay_process
from minidrl.r2d2.run_gym import get_gym_env_creator_fn, gym_model_loader


def run_actor_test(config):
    config.env_creator_fn_getter = get_gym_env_creator_fn
    config.model_loader = gym_model_loader

    mp_ctxt = mp.get_context("spawn")
    terminate_event = mp_ctxt.Event()
    actor_to_replay_queue = mp_ctxt.JoinableQueue(maxsize=config.num_actors * 2)
    learner_to_actor_queue = mp_ctxt.JoinableQueue()
    learner_to_replay_queue = mp_ctxt.JoinableQueue()
    replay_to_learner_queue = mp_ctxt.JoinableQueue()
    log_queue = mp_ctxt.JoinableQueue()

    # spawn replay process
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

    actors = []
    for actor_idx in range(config.num_actors):
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                config,
                actor_to_replay_queue,
                learner_to_actor_queue,
                log_queue,
                terminate_event,
            ),
        )
        actors.append(actor)
        actor.start()

    shared_model = config.load_model()
    shared_model.to(config.actor_device)
    shared_model.share_memory()

    # send shared model params to actor
    for _ in range(config.num_actors):
        learner_to_actor_queue.put(("params", shared_model.state_dict()))

    # give time for actor to add to replay
    learner_to_replay_queue.put(("get_buffer_size", None))
    buffer_size = replay_to_learner_queue.get()
    replay_to_learner_queue.task_done()
    assert buffer_size >= config.learning_starts

    print("main: sending terminate signal")
    terminate_event.set()

    # give time for processes to terminate/stop adding to queues
    time.sleep(2)

    # wait for actors to finish
    learner_to_actor_queue.join()

    print("main: draining queues")
    for i, q in enumerate(
        [
            actor_to_replay_queue,
            learner_to_replay_queue,
            replay_to_learner_queue,
            log_queue,
        ]
    ):
        print(f"main: draining queue {i}")
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
        replay_size=100,
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
        replay_size=8,
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
    for _ in range(1):
        test_run_actor()
    for _ in range(3):
        test_run_multiple_actors()
