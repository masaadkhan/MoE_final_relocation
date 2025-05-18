import torch
import numpy as np
import math
from model import ExpertParallelMoE, Expert
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def swap_experts(moe, idx_a, idx_b):
    # Save current states (as CPU tensors for device-agnostic transfer)
    state_a = {k: v.cpu() for k, v in moe.experts[idx_a].state_dict().items()}
    state_b = {k: v.cpu() for k, v in moe.experts[idx_b].state_dict().items()}
    in_features_a = moe.experts[idx_a].ff[0].in_features
    in_features_b = moe.experts[idx_b].ff[0].in_features

    # Get current devices
    device_a = list(moe.experts[idx_a].parameters())[0].device
    device_b = list(moe.experts[idx_b].parameters())[0].device

    # Re-create on swapped devices
    moe.experts[idx_a] = Expert(in_features_a).to(device_b)
    moe.experts[idx_b] = Expert(in_features_b).to(device_a)

    # Load state dicts on new devices
    moe.experts[idx_a].load_state_dict({k: v.to(device_b) for k, v in state_a.items()})
    moe.experts[idx_b].load_state_dict({k: v.to(device_a) for k, v in state_b.items()})
    print(f"Swapped expert {idx_a} (now on {device_b}) with expert {idx_b} (now on {device_a})")

def main():
    # 1) init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    world_size = torch.cuda.device_count()
    num_experts = world_size

    d_model = 512
    batch_size = 32
    seq_len = 32
    top_k = 1

    expected_tokens = batch_size * seq_len * top_k
    tokens_per_expert = math.ceil(expected_tokens / num_experts)
    expert_capacity = int(tokens_per_expert * 1.25)

    model = ExpertParallelMoE(
        d_model,
        world_size,
        top_k,
        expert_capacity
    ).cuda()

    # 3) wrap *only* router with DDP (so expert weights stay local)
    model.router = DDP(
        model.router,
        device_ids=[local_rank],
        broadcast_buffers=False,
    )

    # 4) optimizer (jointly updates router + experts)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # dummy data
    for step in range(1000):
        x = torch.randn(8, 128, 512, device=local_rank)  # B=8, S=128
        y = torch.randn_like(x)

        out, stats = model(x, return_stats=True)
        # print(f"MASAAD - {stats=}")
        loss = torch.nn.functional.mse_loss(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if local_rank == 0 and step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")

#     stats = np.zeros(world_size, dtype=int)

#     #TODO(MASAAD): Make a formula for the benefit of transferring based on the routing statistics...
#     benefit = 0
#     #TODO(MASAAD): Make a formula for the cost of transferring
#     cost = 1

#     epoch_time_ms = []
#     for epoch in range(num_epochs):
#         stats[:] = 0
#         x = torch.randn(batch_size, seq_len, d_model).to('cuda:0')
#         outputs = moe(x, top_k=top_k, stats=stats)
#         print(f"Epoch {epoch} token distribution:", stats)

#         #TODO(MASAAD): Benefit/Cost calculation should be done in such a way:
#         # Calculate the Benefit of moving Expert A from GPU X
#         # If the Benefit is greater than the Cost of relocating
#         # If 
#         if (benefit > cost):
#             swap_experts(moe, 0, src_gpu=epoch % world_size, dst_gpu=(epoch + 1) % world_size)

if __name__ == "__main__":
    main()
