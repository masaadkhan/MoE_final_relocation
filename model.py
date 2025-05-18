import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        B, S, D = x.shape
        logits = self.gate(x)
        scores = torch.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        dispatch_mask = torch.zeros_like(scores)
        for k in range(self.top_k):
            dispatch_mask.scatter_(-1, topk_indices[..., k:k+1], topk_scores[..., k:k+1])
        return dispatch_mask

class Expert(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class DynamicMoE(nn.Module):
    def __init__(self, d_model, num_experts, expert_devices, top_k):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k=top_k).to("cuda:0")
        self.experts = nn.ModuleList([
            Expert(d_model).to(_) for _ in expert_devices
        ])
        self.expert_devices = list(expert_devices)  # Track current device for each expert

    def forward(self, x, return_stats=False):
        B, S, D = x.shape
        dispatch_mask = self.router(x)
        x_flat = x.view(B*S, D)
        out = torch.zeros_like(x_flat)

        # Stats...
        tokens_per_expert = []
        total_routing_weight = []

        for i, expert in enumerate(self.experts):
            weights = dispatch_mask[..., i].reshape(-1)
            mask = weights > 0

            # Stats...
            tokens_per_expert.append(mask.sum().item())
            total_routing_weight.append(weights.sum().item())

            if mask.sum() == 0:
                continue

            expert_input = x_flat[mask].to(self.expert_devices[i])
            weighted_input = expert_input * weights[mask].to(self.expert_devices[i]).unsqueeze(1)
            expert_output = expert(weighted_input).to(x.device)
            out[mask] = expert_output

        stats = {
            "tokens_per_expert": tokens_per_expert,
            "routing_weight_per_expert": total_routing_weight,
            "max_tokens": max(tokens_per_expert),
            "most_used_expert": int(torch.tensor(tokens_per_expert).argmax()),
        }

        if return_stats:
            return out.view(B, S, D), stats
        else:
            return out.view(B, S, D)
