"""Playing around with torch shared memory."""
import time

import torch
import torch.multiprocessing as mp
from torch import nn, optim

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

    def forward(self, x):
        return self.net(x)


def worker_fn(
    worker_id,
    learner_send_queue,
    learner_recv_queue,
    replay_queue,
    terminate_event,
):
    """Worker function."""
    print(f"Worker {worker_id} starting")

    worker_model = SharedNet()
    worker_model.cpu()
    for param in worker_model.parameters():
        param.requires_grad = False

    model_output_weight = worker_model.net[2].weight[0]

    print(f"Worker {worker_id} requesting reference to learner model weights")
    learner_send_queue.put(("get_latest_params", worker_id))
    learner_params = learner_recv_queue.get()
    learner_recv_queue.task_done()

    step = 0
    while not terminate_event.is_set():
        if step % 400 == 0:
            print(f"Worker {worker_id} {step=} syncing model weights")
            worker_model.load_state_dict(learner_params)

            model_output_weight = worker_model.net[2].weight[0]
            if worker_id == 0:
                print(f"Worker {worker_id} weights={model_output_weight}")

        input = torch.rand(INPUT_SIZE)
        output = worker_model(input)
        replay_queue.put((input, output))
        step += 1

    del learner_params

    # learner_send_queue.join()
    replay_queue.join()

    print(f"Worker {worker_id} done")


def main():
    """Main function."""
    num_workers = 2

    # `fork` not supported by CUDA
    # https://pytorch.org/docs/main/notes/multiprocessing.html#cuda-in-multiprocessing
    # must use context to set start method
    mp_ctxt = mp.get_context("spawn")

    terminate_event = mp_ctxt.Event()
    replay_queue = mp_ctxt.JoinableQueue()
    learner_send_queue = [mp_ctxt.JoinableQueue() for _ in range(num_workers)]
    learner_recv_queue = mp_ctxt.JoinableQueue()

    workers = []
    for i in range(num_workers):
        worker = mp_ctxt.Process(
            target=worker_fn,
            args=(
                i,
                learner_recv_queue,
                learner_send_queue[i],
                replay_queue,
                terminate_event,
            ),
        )
        worker.start()
        workers.append(worker)

    main_model = SharedNet()
    main_model.share_memory()
    main_model.cuda()

    optimizer = optim.Adam(main_model.parameters(), lr=1e-3)

    for i in range(10):
        print(f"\nIteration {i}")

        batch = []
        print("Boss: collecting next batch")
        while len(batch) < 10:
            if not learner_recv_queue.empty():
                request = learner_recv_queue.get()
                print(f"Boss: sending model weights to worker {request[1]}")
                learner_send_queue[request[1]].put(main_model.state_dict())
                learner_recv_queue.task_done()

            if not replay_queue.empty():
                batch.append(replay_queue.get()[0])
                replay_queue.task_done()

        print("Boss: batch received, updating")
        batch = torch.stack(batch).cuda()
        outputs = main_model(batch)
        targets = torch.ones_like(outputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Boss: iteration={i} loss={loss:.4f}")

    print("Boss done")
    terminate_event.set()
    time.sleep(1)

    print("Boss: closing replay queue")
    while not replay_queue.empty():
        replay_queue.get()
        replay_queue.task_done()
    replay_queue.close()

    print("Boss: closing learner request queue")
    while not learner_recv_queue.empty():
        learner_recv_queue.get()
        learner_recv_queue.task_done()
    learner_recv_queue.close()

    print("Boss: closing learner send queue")
    for i in range(num_workers):
        learner_send_queue[i].join()
        learner_send_queue[i].close()

    print("Stop signal sent, joining workers")
    for i in range(num_workers):
        workers[i].join()

    print("All done")


if __name__ == "__main__":
    main()
