import torch
import torch.nn as nn
from model import DynamicMoE
import time

NVLINK4_BW_GBPS = 450
def calc_benefit(number_of_tokens, token_dtype, d_model):
    # Amount of time to transfer tokens
    return (number_of_tokens * token_dtype * d_model) / (NVLINK4_BW_GBPS * 1e9)

# You have to do two sequential expert copies from A to B and then B to A
# You could make this "in parallel" if you stored a tmp on CPU or something to that extent
def calc_cost(expert_size_bytes):
    return (2 * expert_size_bytes) / (NVLINK4_BW_GBPS * 1e9)

def calc_module_bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())

def train_loop(swap_experts=False):
    cost_scale = 1.25
    decay = 1
    router_gpu = 0
    accum_benefit = torch.zeros(num_experts, dtype=torch.float32, device='cuda')
    swap_count = 0
    cooldown_steps = 5

    train_times = []
    total_steps = 10000

    start_loop_time = time.perf_counter()
    cooldown = cooldown_steps
    for step in range(total_steps):
        # x = torch.ones(B, S, d_model, device="cuda:0")
        x = torch.randn(B, S, d_model, device="cuda:0")
        if step < 300:
            x[..., :8] += 10
            x[..., :8:16] += 100
            x[..., :16:24] += 1000
        elif step < 600:
            x[..., :8] += 50
            x[..., :8:16] += 500
            x[..., :16:24] += 5000
        elif step < 900:
            x[..., :8] += 100
            x[..., :8:16] += 1000
            x[..., :16:24] += 10000
        
        torch.cuda.synchronize()
        train_start_time = time.perf_counter()

        out, tokens_per_expert = moe(x, return_stats=True)
        y = torch.randn_like(out)
        loss = nn.MSELoss()(out, y)
        loss.backward()

        # Screwed over because of DDP autograd problem.....
        # optimizer.step()
        # optimizer.zero_grad()

        torch.cuda.synchronize()
        train_elapsed_time = time.perf_counter() - train_start_time
        train_times.append(train_elapsed_time)

        # print(f"Step {step}: loss={loss.item():.4f}")

        router_tokens = tokens_per_expert[moe.experts[router_gpu].id]
        benefit = calc_benefit(tokens_per_expert - router_tokens, 4, d_model) # Assuming FP32...
        accum_benefit = (accum_benefit + benefit).clamp(min=0)

        # Amount of time to transfer an expert with size bytes
        cost_to_copy = calc_cost(calc_module_bytes(moe.experts[router_gpu]))
        max_accum_value, max_idx = accum_benefit.max(dim=0)

        tokens_per_expert_float = tokens_per_expert.float()
        mean_of_tokens = torch.mean(tokens_per_expert_float)
        std_of_tokens = torch.std(tokens_per_expert_float)
        var_of_tokens = torch.var(tokens_per_expert_float)

        # print(f"{mean_of_tokens=}")
        # print(f"{std_of_tokens=}")
        # print(f"{var_of_tokens=}")

        # print(f"{benefit=}")
        # print(f"{accum_benefit=}")
        # print(f"{cost_to_copy=}")
        # print(f"{max_accum_value=}")
        # print(f"{max_idx=}")
        # print("")

        if swap_experts:
            # print(f"{cooldown=}")
            if (max_accum_value > (cost_to_copy * cost_scale)) and cooldown == 0:
                # When the router's expert lines up with the max token accumulation error
                if (moe.experts[router_gpu].id == max_idx):
                    print(f"{router_gpu=} == {max_idx=}")

                    print(f"{benefit=}")
                    print(f"{accum_benefit=}")
                    print(f"{cost_to_copy=}")
                    print(f"{max_accum_value=}")
                    print(f"{max_idx=}")
                    print("")
                    exit()

                moe.swap_experts(moe.experts[router_gpu].id, moe.experts[max_idx].id)
                accum_benefit.zero_()
                # print(f"{accum_benefit=} after swap...")
                swap_count += 1
                cooldown = 5
            else:
                accum_benefit *= decay
                cooldown = max(0, cooldown - 1)

    elapsed_loop_time = time.perf_counter() - start_loop_time
    print(f"Total loop took {elapsed_loop_time} ms")
    print(f"Swapped {swap_count} amount of times during this training job...")
    return train_times

if __name__ == "__main__":
    d_model = 512
    top_k = 1
    B = 128
    S = 512

    num_experts = torch.cuda.device_count()

    assert (B % num_experts) == 0, "Batch size should be a multiple of number of experts..."
    assert (S % num_experts) == 0, "Sequence Len should be a multiple of number of experts..."

    devices = [f"cuda:{i}" for i in range(num_experts)]

    moe = DynamicMoE(d_model, num_experts, devices, top_k)
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
    expert_alloc = [_.id for _ in moe.experts]

    times = train_loop()
    avg_time = sum(times) / len(times)
    print(f"Average time to finish train loop (no swap): {avg_time}")

    times = train_loop(swap_experts=True)
    avg_time = sum(times) / len(times)
    print(f"Average time to finish train loop (swap): {avg_time}")
