import time

import torch.multiprocessing as mp


def run_actor(terminate_event: mp.Event):
    print("Actor process started")
    while not terminate_event.is_set():
        time.sleep(1)


def run_replay(terminate_event: mp.Event):
    print("Replay process started")
    start_time = time.time()
    while not terminate_event.is_set() and time.time() - start_time < 5:
        time.sleep(1)

    raise Exception("Replay process failed")


def run_learner(terminate_event: mp.Event):
    print("Learner process started")
    start_time = time.time()
    while not terminate_event.is_set() and time.time() - start_time < 30:
        time.sleep(1)

    return


def main():
    mp_ctxt = mp.get_context("spawn")

    terminate_training_event = mp_ctxt.Event()

    actor = mp_ctxt.Process(target=run_actor, args=(terminate_training_event,))
    replay = mp_ctxt.Process(target=run_replay, args=(terminate_training_event,))
    learner = mp_ctxt.Process(target=run_learner, args=(terminate_training_event,))
    actor.start()
    replay.start()
    learner.start()

    while True:
        time.sleep(1)
        if not actor.is_alive():
            print("Actor process failed")
            break
        if not replay.is_alive():
            print("Replay process failed")
            break
        if not learner.is_alive():
            print("Learner process failed")
            break

    terminate_training_event.set()
    print("Joining actor process")
    actor.join()
    print("Joining replay process")
    replay.join()
    print("Joining learner process")
    learner.join()
    print("Main process exiting")


if __name__ == "__main__":
    main()
