import torch
import torch.nn as nn
from model import DynamicMoE

NVLINK4_BW_GBPS = 450
def calc_benefit(number_of_tokens, token_dtype, d_model):
    # Amount of time to transfer tokens
    return (number_of_tokens * token_dtype * d_model) / (NVLINK4_BW_GBPS * 1e9)

def calc_cost(expert_size_bytes):
    return expert_size_bytes / (NVLINK4_BW_GBPS * 1e9)

def calc_module_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())

if __name__ == "__main__":
    d_model = 512
    #TODO(MASAAD): Can artificially increase the amount of expert traffic increase this...
    top_k = 4
    B = 128
    S = 512

    num_experts = torch.cuda.device_count()

    assert (B % num_experts) == 0, "Batch size should be a multiple of number of experts..."
    assert (S % num_experts) == 0, "Sequence Len should be a multiple of number of experts..."

    devices = [f"cuda:{i}" for i in range(num_experts)]

    moe = DynamicMoE(d_model, num_experts, devices, top_k)
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
    expert_alloc = [_.id for _ in moe.experts]

    k = 1.25
    decay = 0.95
    router_gpu = 0
    accum_benefit = torch.zeros(num_experts, dtype=torch.float32, device='cuda')

    total_steps = 100
    for step in range(total_steps):
        x = torch.randn(B, S, d_model, device="cuda:0")
        out, tokens_per_expert = moe(x, return_stats=True)
        y = torch.randn_like(out)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {step}: loss={loss.item():.4f}")

        router_tokens = tokens_per_expert[moe.experts[0].id]
        benefit = calc_benefit(tokens_per_expert - router_tokens, 4, d_model) # Assuming FP32...
        accum_benefit = (accum_benefit + benefit).clamp(min=0)

        # benefit_cpu = benefit.detach().cpu().tolist()
        # Amount of time to transfer an expert with size bytes
        cost_to_copy = calc_cost(calc_module_bytes(moe.experts[0]))

        max_accum_value, max_idx = accum_benefit.max(dim=0)

        print(f"{benefit=}")
        print(f"{accum_benefit=}")
        print(f"{cost_to_copy=}")
        print(f"{max_accum_value=}")
        print(f"{max_idx=}")
        print("")

        if max_accum_value > (cost_to_copy * k):
            moe.swap_experts(moe.experts[router_gpu].id, moe.experts[max_idx].id)
        else:
            accum_benefit *= decay

        # exit()
