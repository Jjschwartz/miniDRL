import threading
import time
from multiprocessing.queues import Empty

import torch
import torch.multiprocessing as mp


model_dims = (10, 10)


def run_actor(actor_idx: int, send_queue: mp.Queue, recv_queue: mp.Queue):
    """Actor function using queues for interprocess communication."""
    print(f"actor={actor_idx}: Actor started")
    params = torch.randn(model_dims, dtype=torch.float32)

    t = 0
    while True:
        try:
            recv_queue.get_nowait()
            print(f"actor={actor_idx} - {t=}: End training signal recieved.")
            break
        except Empty:
            pass
        # do work
        time.sleep(0.01)

        if t % 100 == 0:
            print(f"actor={actor_idx} - {t=}: syncing")
            send_queue.put(("sync", actor_idx))
            result = recv_queue.get()
            # need to handle case where exit signal has been sent in the meantime
            if result[0] == "params":
                params = result[1]
                print(f"actor={actor_idx} - {t=}: synced params = {params.mean()}")
            else:
                assert result[0] == "end_training"
                print(f"actor={actor_idx} - {t=}: End training signal recieved.")

        if actor_idx == 0 and t % 50 == 0:
            print(f"actor={actor_idx} - {t=}: sending results")
            send_queue.put(("results", {"t": t}))

        t += 1

    del params

    print(f"actor={actor_idx} - {t=}: done")


def main():
    num_actors = 2

    model_params = torch.zeros(model_dims, dtype=torch.float32)

    mp_ctxt = mp.get_context("spawn")
    actor_recv_queue = mp_ctxt.Queue()
    actor_send_queues = [mp_ctxt.Queue() for _ in range(num_actors)]

    actors = []
    for actor_idx in range(num_actors):
        actor = mp_ctxt.Process(
            target=run_actor,
            args=(
                actor_idx,
                actor_recv_queue,
                actor_send_queues[actor_idx],
            ),
        )
        actor.start()
        actors.append(actor)

    end_training = threading.Event()

    def run_param_sync_thread():
        print(f"Param sync thread - step={global_step}: started")
        num_syncs = 0
        num_results = 0
        while not end_training.is_set():
            try:
                # set timeout so that we can check if training is done
                request = actor_recv_queue.get(timeout=1)
                if request[0] == "sync":
                    num_syncs += 1
                    print(
                        f"Param sync thread - step={global_step}: "
                        f"sync {num_syncs} = {request[1]}"
                    )
                    actor_send_queues[request[1]].put(("params", model_params))
                elif request[0] == "results":
                    num_results += 1
                    print(
                        f"Param sync thread - step={global_step}: "
                        f"result {num_results} ={request[1]}"
                    )
                else:
                    raise ValueError(f"Unknown request {request}")
            except Empty:
                pass

        print(
            f"Param sync thread - step={global_step}: finished, "
            f"{num_syncs=}, {num_results=}"
        )

    param_sync_thread = threading.Thread(target=run_param_sync_thread)
    param_sync_thread.start()

    print("main: training loop started")
    global_step = 0
    while global_step < 3:
        time.sleep(2)
        model_params += 1
        print(f"main: step={global_step} complete, params={model_params.mean()}")
        global_step += 1

    print("main: sending end training signal")
    end_training.set()
    print("main: waiting for param sync thread to finish")
    param_sync_thread.join()

    print("main: sending end training signal to actors")
    for idx in range(num_actors):
        actor_send_queues[idx].put(("end_training",))

    print("main: waiting for actors to finish")
    for actor in actors:
        actor.join()

    print("main: emptying actor queues")
    for q in actor_send_queues:
        while not q.empty():
            q.get()

    while not actor_recv_queue.empty():
        actor_recv_queue.get()

    print("main: done")


if __name__ == "__main__":
    main()
