import torch
import torch.nn as nn
from model import DynamicMoE, Expert

def swap_experts(moe, idx_a, idx_b):
    state_a = {k: v.cpu() for k, v in moe.experts[idx_a].state_dict().items()}
    state_b = {k: v.cpu() for k, v in moe.experts[idx_b].state_dict().items()}
    in_features_a = moe.experts[idx_a].ff[0].in_features
    in_features_b = moe.experts[idx_b].ff[0].in_features

    device_a = list(moe.experts[idx_a].parameters())[0].device
    device_b = list(moe.experts[idx_b].parameters())[0].device

    moe.experts[idx_a] = Expert(in_features_a).to(device_b)
    moe.experts[idx_b] = Expert(in_features_b).to(device_a)

    moe.experts[idx_a].load_state_dict({k: v.to(device_b) for k, v in state_a.items()})
    moe.experts[idx_b].load_state_dict({k: v.to(device_a) for k, v in state_b.items()})
    print(f"Swapped expert {idx_a} (now on {device_b}) with expert {idx_b} (now on {device_a})")

if __name__ == "__main__":
    d_model = 512
    top_k = 2
    B = 8
    S = 32

    num_experts = torch.cuda.device_count()

    assert (B % num_experts) == 0

    devices = [f"cuda:{i}" for i in range(num_experts)]

    moe = DynamicMoE(d_model, num_experts, devices, top_k)
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)

    for epoch in range(5):
        x = torch.randn(B, S, d_model, device="cuda:0")
        out, stats = moe(x, return_stats=True)
        # print(f"{stats=}")
        y = torch.randn_like(out)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
