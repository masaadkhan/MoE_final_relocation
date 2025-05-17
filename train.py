import torch
import numpy as np
import math
from model import ExpertParallelMoE, Expert

# TODO(MASAAD): Understand this code better...
# Seems like we're taking the src_expert
# We create a dictionary keeping the expert_id? and v the actual expert weights?
# We create an expert object and send it to the dst gpu
# We load_state_dict? Not sure what that does right now
# Sends that to dst_gpu
# And apparently this is "migrating an expert"
# From my understanding this should be like a tmp data swap...
# tmp = src
# src = dst
# dst = tmp
def migrate_expert(moe, expert_idx, src_gpu, dst_gpu):
    src_expert = moe.experts[expert_idx].to(f'cuda:{src_gpu}')
    expert_state = {k: v.cpu() for k, v in src_expert.state_dict().items()}
    moe.experts[expert_idx] = Expert(src_expert.ff[0].in_features).to(f'cuda:{dst_gpu}')
    moe.experts[expert_idx].load_state_dict({k: v.to(f'cuda:{dst_gpu}') for k, v in expert_state.items()})
    print(f"Expert {expert_idx} migrated from GPU {src_gpu} to GPU {dst_gpu}.")

def train_moe(num_epochs=5, top_k=1):
    world_size = torch.cuda.device_count()
    num_experts = world_size
    print(f"World size or num_experts: {world_size}")

    d_model = 512
    batch_size = 32
    seq_len = 32

    expected_tokens = batch_size * seq_len * top_k
    tokens_per_expert = expected_tokens / num_experts
    expert_capacity = int(tokens_per_expert * 1.25)

    moe = ExpertParallelMoE(d_model, world_size, expert_capacity).to('cuda:0')
    stats = np.zeros(world_size, dtype=int)

    #TODO(MASAAD): Make a formula for the benefit of transferring based on the routing statistics...
    benefit = 0
    #TODO(MASAAD): Make a formula for the cost of transferring
    cost = 1

    epoch_time_ms = []
    for epoch in range(num_epochs):
        stats[:] = 0
        x = torch.randn(batch_size, seq_len, d_model).to('cuda:0')
        outputs = moe(x, top_k=top_k, stats=stats)
        print(f"Epoch {epoch} token distribution:", stats)

        if (benefit > cost):
            migrate_expert(moe, 0, src_gpu=epoch % world_size, dst_gpu=(epoch + 1) % world_size)

if __name__ == "__main__":
    train_moe(num_epochs=5, top_k=1)
