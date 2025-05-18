import torch
import torch.nn as nn
import torch.distributed as dist

class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class Router(nn.Module):
    def __init__(self, d_model, top_k, total_num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, total_num_experts)
        self.top_k = top_k
        self.total_num_experts = total_num_experts
        assert self.top_k <= self.total_num_experts, f"{top_k=} can't exceed number of experts={self.total_num_experts}"

    def forward(self, x):
        logits = self.gate(x)
        gates = torch.softmax(logits, dim=-1)

        # pick top-k gates + indices
        topk_vals, topk_idx = gates.topk(self.top_k, dim=-1)
        
        # build the dispatch matrix
        dispatch_mask = torch.zeros_like(gates)
        dispatch_mask.scatter_(1, topk_idx, topk_vals)
        
        return dispatch_mask, topk_idx

class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, top_k, num_experts_per_gpu, expert_capacity):
        super().__init__()
        self.world_size            = dist.get_world_size()
        self.rank                  = dist.get_rank()
        self.num_experts_per_gpu   = num_experts_per_gpu
        self.total_num_experts     = self.world_size * num_experts_per_gpu
        self.expert_capacity       = expert_capacity
        self.top_k                 = top_k

        self.router = Router(d_model, top_k, self.total_num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model) for _ in range(num_experts_per_gpu)]
        )

    def forward(self, x, return_stats: bool = False):
        B, S, D  = x.shape
        T        = B * S
        x_flat   = x.view(T, D)                     # [T, D]

        # ── 1) ROUTING ────────────────────────────────
        # Router returns (dispatch_mask [T,N], topk_idx [T,top_k])
        dispatch_mask, topk_idx = self.router(x_flat)

        # === COMPUTE STATS BEFORE YOU BUCKET ===
        #  global expert counts
        tok_per_exp = dispatch_mask.sum(dim=0)         # [N]
        #  reshape into [G, E]
        counts      = tok_per_exp.view(self.world_size,
                                       self.num_experts_per_gpu)
        #  per-GPU totals
        tok_per_gpu = counts.sum(dim=1)                # [G]

        stats = {
            "tokens_per_expert":    tok_per_exp.cpu().tolist(),
            "tokens_per_gpu":       tok_per_gpu.cpu().tolist(),
            "mean_tokens":          tok_per_exp.float().mean().item(),
            "var_tokens":           tok_per_exp.float().var(unbiased=False).item(),
            "max_tokens":           tok_per_exp.max().item(),
            "min_tokens":           tok_per_exp.min().item(),
            "num_capacity_hits":    int((tok_per_exp > self.expert_capacity).sum().item()),
        }

        # ── 2) SPLIT INTO PER-GPU BUCKETS ───────────────
        buckets_x, buckets_mask = [], []
        E = self.num_experts_per_gpu
        for g in range(self.world_size):
            start, end = g * E, (g + 1) * E
            mask_g     = dispatch_mask[:, start:end]  # [T, E]
            token_sel  = mask_g.sum(dim=1) > 0        # [T]
            idx        = token_sel.nonzero(as_tuple=True)[0]

            buckets_x.append(   x_flat[idx]   )       # [Lg, D]
            buckets_mask.append(mask_g[idx])          # [Lg, E]

        # pad to same length
        L_max = max(b.size(0) for b in buckets_x)
        for i in range(self.world_size):
            L = buckets_x[i].size(0)
            if L < L_max:
                pad_x = torch.zeros(L_max - L, D, device=x.device)
                buckets_x[i]   = torch.cat([buckets_x[i],   pad_x], dim=0)
                pad_m = torch.zeros(L_max - L, E, device=x.device)
                buckets_mask[i] = torch.cat([buckets_mask[i], pad_m], dim=0)

        # ── 3) ALL-TO-ALL DISPATCH ───────────────────────
        recv_x    = [torch.zeros_like(buckets_x[0])    for _ in range(self.world_size)]
        recv_mask = [torch.zeros_like(buckets_mask[0]) for _ in range(self.world_size)]
        dist.all_to_all(recv_x,    buckets_x)
        dist.all_to_all(recv_mask, buckets_mask)

        # ── 4) APPLY LOCAL EXPERTS ───────────────────────
        out_buckets = []
        for _g in range(self.world_size):
            x_in    = recv_x[_g]       # [L_max, D]
            mask_in = recv_mask[_g]    # [L_max, E]
            o       = torch.zeros_like(x_in)
            # each local expert j ∈ [0, E)
            for j, expert in enumerate(self.experts):
                w = mask_in[:, j].unsqueeze(1)   # [L_max,1]
                if w.any():
                    o = o + expert(x_in) * w
            out_buckets.append(o)               # [L_max, D]

        # ── 5) ALL-TO-ALL GATHER BACK ────────────────────
        gather_out = [torch.zeros_like(out_buckets[0]) for _ in range(self.world_size)]
        dist.all_to_all(gather_out, out_buckets)

        # ── 6) UNBUCKET & REASSEMBLE ────────────────────
        out_flat = torch.zeros(T, D, device=x.device)
        for g in range(self.world_size):
            start, end = g * E, (g + 1) * E
            mask_g     = dispatch_mask[:, start:end]
            idx        = (mask_g.sum(dim=1) > 0).nonzero(as_tuple=True)[0]
            out_flat[idx] += gather_out[g][: idx.numel()]

        if return_stats:
            return out_flat.view(B, S, D), stats
        else:
            return out_flat.view(B, S, D)
