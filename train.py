import torch
import numpy as np
from model import DynamicMoE, Expert
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

if __name__ == "__main__":
    d_model = 512
    num_experts = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_experts)]

    moe = DynamicMoE(d_model, num_experts, devices)
    for i, expert in enumerate(moe.experts):
        print(f"BEFORE Expert {i}: {next(expert.parameters()).device}")
    print("Router device:", next(moe.router.parameters()).device)
    moe = moe.to("cuda:0")  # router/main on cuda:0

    for i, expert in enumerate(moe.experts):
        print(f"AFTER Expert {i}: {next(expert.parameters()).device}")
    print("Router device:", next(moe.router.parameters()).device)

    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)

    B, S = 8, 32
    for epoch in range(5):
        x = torch.randn(B, S, d_model, device="cuda:0")
        # Randomly assign each token to an expert
        assignments = torch.randint(0, num_experts, (B*S,), device="cuda:0")
        out = moe(x, assignments)
        y = torch.randn_like(out)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

        # Example: After epoch 2, relocate expert 0 to GPU 1
        if epoch == 2 and num_experts > 1:
            # moe.relocate_expert(0, "cuda:1")
            # Recreate optimizer to keep params on correct device
            optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
