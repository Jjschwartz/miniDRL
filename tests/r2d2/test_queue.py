"""Tests for R2D2 replay buffer."""
import time
from multiprocessing.queues import Empty

import torch
import torch.multiprocessing as mp


T = 120
B = 4
C = 40
iters = 1000


def run_queue_worker(idx, input_queue, output_queue):
    t = 0
    while True:
        try:
            input_queue.get_nowait()
            print(f"i={idx} - {t=}: End signal recieved.")
            break
        except Empty:
            pass

        obs_seq_buffer = torch.randn((T, B, 4), dtype=torch.float32)
        action_seq_buffer = torch.randint(0, 3, (T, B), dtype=torch.long)
        reward_seq_buffer = torch.randn((T, B), dtype=torch.float32)
        done_buffer = torch.randint(0, 2, (T, B), dtype=torch.long)
        lstm_h_seq_buffer = torch.randn((1, B, 64), dtype=torch.float32)
        lstm_c_seq_buffer = torch.randn((1, B, 64), dtype=torch.float32)
        priorities = torch.randn((B,), dtype=torch.float32)

        output_queue.put(
            (
                obs_seq_buffer,
                action_seq_buffer,
                reward_seq_buffer,
                done_buffer,
                lstm_h_seq_buffer,
                lstm_c_seq_buffer,
                priorities,
                idx,
            )
        )
        t += 1

    output_queue.close()
    input_queue.get()
    print(f"i={idx} - {t=}: Ended.")


def test_queue():
    print("main: start")
    mp_ctxt = mp.get_context("spawn")
    input_queue = mp_ctxt.Queue()
    output_queue = mp_ctxt.Queue(maxsize=30)
    worker = mp_ctxt.Process(
        target=run_queue_worker,
        args=(0, input_queue, output_queue),
    )
    worker.start()

    obs_seq_buffer = torch.zeros((T, C, 4), dtype=torch.float32)
    action_seq_buffer = torch.zeros((T, C), dtype=torch.long)
    reward_seq_buffer = torch.zeros((T, C), dtype=torch.float32)
    done_buffer = torch.zeros((T, C), dtype=torch.long)
    lstm_h_seq_buffer = torch.zeros((1, C, 64), dtype=torch.float32)
    lstm_c_seq_buffer = torch.zeros((1, C, 64), dtype=torch.float32)
    priorities = torch.randn((C,), dtype=torch.float32)

    print("main: starting loop")
    start_time = time.time()
    for t in range(iters):
        index = (t * B) % C
        batch = output_queue.get()
        obs_seq_buffer[:, index : index + B] = batch[0]
        action_seq_buffer[:, index : index + B] = batch[1]
        reward_seq_buffer[:, index : index + B] = batch[2]
        done_buffer[:, index : index + B] = batch[3]
        lstm_h_seq_buffer[:, index : index + B] = batch[4]
        lstm_c_seq_buffer[:, index : index + B] = batch[5]
        priorities[index : index + B] = batch[6]

    print("main: loop finished")
    time_taken = time.time() - start_time
    print(f"main: time per step = {time_taken / iters:.6f}")

    print("main: sending end signal")
    input_queue.put(0)

    print("main: flushing output queue")
    try:
        while True:
            output_queue.get_nowait()
    except Empty:
        pass

    print("main: flushing complete.")
    input_queue.put(0)

    print("main: waiting for worker to end")
    input_queue.close()
    output_queue.close()
    worker.join()
    print("main: closing queues")

    print("main: end")


if __name__ == "__main__":
    test_queue()
