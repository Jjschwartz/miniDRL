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


def worker_fn(
    worker_id, model, input_queue, shared_input_array, output_queue, shared_output_array
):
    """Worker function."""
    print(f"Worker {worker_id} starting")

    worker_model = SharedNet()
    worker_model.cpu()

    while True:
        print(f"Worker {worker_id} syncing model weights")
        worker_model.load_state_dict(model.state_dict())

        do_work = input_queue.get()
        if do_work == 0:
            break

        work_to_do = shared_input_array[:]
        print(f"Worker {worker_id} work to do: {work_to_do}")
        time.sleep(1)
        work_output = torch.randint(0, 10, size=(1,))
        print(f"Worker {worker_id} work done output = {work_output}")
        shared_output_array[:] = work_output

        output_queue.put(1)

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

    shared_input_array = torch.zeros(
        [num_workers, 3], dtype=torch.int32
    ).share_memory_()
    shared_output_array = torch.zeros(
        [num_workers, 1],
        dtype=torch.int32,
    ).share_memory_()

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
                shared_input_array[i],
                output_queues[i],
                shared_output_array[i],
            ),
        )
        worker.start()
        workers.append(worker)

    for i in range(3):
        print(f"\nIteration {i}")
        for j in range(num_workers):
            work_to_do = torch.randint(0, 10, size=(3,))
            print(f"Boss: worker {j} work to do: {work_to_do}")
            shared_input_array[j] = work_to_do

        print("Boss: Telling workers that work is ready")
        for j in range(num_workers):
            input_queues[j].put(1)

        print("Boss waiting: for work to be done")
        for j in range(num_workers):
            output_queues[j].get()

        print("Boss: all work done")
        print(f"Shared output array: {shared_output_array[:]}")
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
