"""Playing around with torch shared memory."""
import time

import torch
import torch.multiprocessing as mp
from torch import nn


INPUT_SIZE = 3
OUTPUT_SIZE = 1


class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 4),
            nn.ReLU(),
            nn.Linear(4, OUTPUT_SIZE),
        )


def worker_fn(worker_id, model, input_queue, output_queue):
    """Worker function."""
    print(f"Worker {worker_id} starting")

    worker_model = SharedNet()
    worker_model.cpu()

    while True:
        print(f"Worker {worker_id} syncing model weights")
        worker_model.load_state_dict(model.state_dict())

        work = input_queue.get()
        if isinstance(work, int):
            break

        print(f"Worker {worker_id} work to do: {work}")
        time.sleep(1)
        work[:] = torch.randint(0, 10, size=work.shape)
        print(f"Worker {worker_id} work done output = {work}")

        output_queue.put(work)

    print(f"Worker {worker_id} done")


def main():
    """Main function."""
    num_workers = 2

    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    main_model = SharedNet()
    main_model.share_memory()
    main_model.cuda()

    shared_array = torch.zeros([num_workers, 3], dtype=torch.int32).share_memory_()

    input_queues, output_queues = [], []
    for i in range(num_workers):
        input_queues.append(mp_ctxt.Queue())
        output_queues.append(mp_ctxt.Queue())

    workers = []
    for i in range(num_workers):
        worker = mp_ctxt.Process(
            target=worker_fn,
            args=(
                i,
                main_model,
                input_queues[i],
                output_queues[i],
            ),
        )
        worker.start()
        workers.append(worker)

    for i in range(2):
        print(f"\nIteration {i}")
        for j in range(num_workers):
            shared_array[j] = torch.randint(0, 10, size=(3,))
            work_to_do = shared_array[j]
            print(f"Boss: worker {j} work to do: {work_to_do}")
            print(f"Boss: worker {j} shared array: {shared_array[j]}")
            input_queues[j].put(work_to_do)

        print("Boss waiting: for work to be done")
        for j in range(num_workers):
            work_done = output_queues[j].get()
            print(f"Boss: worker {j} work done: {work_done}")
            print(f"Boss: worker {j} shared array: {shared_array[j]}")
            del work_done

        print("Boss: all work done")
        print(f"Shared array: {shared_array[:]}")
        print("Boss: thinking/updating model...")
        time.sleep(1)

    print("Work over, sendin stop signal")
    for i in range(num_workers):
        input_queues[i].put(0)

    print("Stop signal sent, joining workers")
    for i in range(num_workers):
        workers[i].join()

    for i in range(num_workers):
        input_queues[i].close()
        output_queues[i].close()

    print("All done")


if __name__ == "__main__":
    main()
