"""Benchmarking using pipes versus two queues for synced interprocess communication."""
import time

import torch
import torch.multiprocessing as mp


def actor_queue(input_queue: mp.Queue, output_queue: mp.Queue):
    """Actor function using queues for interprocess communication."""
    print("Actor started")
    while True:
        work = input_queue.get()
        if work is None:
            break

        result = torch.randn(work.shape, dtype=torch.float32)
        output_queue.put(result)

    print("Actor done")


def actor_pipe(pipe: mp.Pipe):
    """Actor function using pipes for interprocess communication."""
    print("Actor started")
    while True:
        work = pipe.recv()
        if work is None:
            break

        result = torch.randn(work.shape, dtype=torch.float32)
        pipe.send(result)

    print("Actor done")


def main():
    num_iterations = 10000
    num_actors = 2

    mp_ctxt = mp.get_context("spawn")

    queue_actors = []
    input_queues, output_queues = [], []
    for actor_idx in range(num_actors):
        input_queue = mp_ctxt.Queue()
        output_queue = mp_ctxt.Queue()
        queue_actors.append(
            mp_ctxt.Process(
                target=actor_queue,
                args=(input_queue, output_queue),
            )
        )
        input_queues.append(input_queue)
        output_queues.append(output_queue)

    for actor in queue_actors:
        actor.start()

    start_time = time.time()
    for i in range(num_iterations):
        for j in range(num_actors):
            input_queues[j].put(torch.randn((3,), dtype=torch.float32))

        for j in range(num_actors):
            output_queues[j].get()

    for j in range(num_actors):
        input_queues[j].put(None)

    for actor in queue_actors:
        actor.join()

    for j in range(num_actors):
        input_queues[j].close()
        output_queues[j].close()

    print(f"Queue time: {time.time() - start_time}")

    pipe_actors = []
    pipes = []
    for actor_idx in range(num_actors):
        main_conn, actor_conn = mp_ctxt.Pipe()
        pipe_actors.append(
            mp_ctxt.Process(
                target=actor_pipe,
                args=(actor_conn,),
            )
        )
        pipes.append((main_conn, actor_conn))

    for actor in pipe_actors:
        actor.start()

    start_time = time.time()
    for i in range(num_iterations):
        for j in range(num_actors):
            pipes[j][0].send(torch.randn((3,), dtype=torch.float32))

        for j in range(num_actors):
            pipes[j][0].recv()

    for j in range(num_actors):
        pipes[j][0].send(None)

    for actor in pipe_actors:
        actor.join()

    for j in range(num_actors):
        pipes[j][0].close()
        pipes[j][1].close()

    print(f"Pipe time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
