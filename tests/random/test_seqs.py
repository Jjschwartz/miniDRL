import torch

num_rollout_steps = 8
num_envs = 2
seq_len = 4
num_seqs_per_rollout = num_rollout_steps // seq_len
buf_shape = (num_rollout_steps, num_envs)

obs = torch.zeros(buf_shape)
for i in range(num_envs):
    for t in range(num_rollout_steps):
        obs[t, i] = i * num_rollout_steps + t

print(f"{obs.shape=}")
print(obs)

b_obs = torch.concatenate(torch.split(obs, seq_len, dim=0), dim=1)
print(f"{b_obs.shape=}")
print(b_obs)

lstm_state = torch.zeros((num_seqs_per_rollout,) + (1, num_envs, 3))
for t in range(num_seqs_per_rollout):
    for i in range(num_envs):
        lstm_state[t, 0, i, :] = t * seq_len + i

print(f"{lstm_state.shape=}")
print(lstm_state)

b_lstm_states = lstm_state.view(1, num_envs * num_seqs_per_rollout, 3)
print(f"{b_lstm_states.shape=}")
print(b_lstm_states)

print()
for b_idx in range(num_envs * num_seqs_per_rollout):
    print(f"b_idx={b_idx}")
    print(f"{b_obs[:, b_idx]=}")
    print(f"{b_lstm_states[0, b_idx, :]=}")
