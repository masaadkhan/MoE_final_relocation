import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        logits = self.gate(x)  # [B, S, num_experts]
        # Softmax over experts
        scores = torch.softmax(logits, dim=-1)  # [B, S, num_experts]
        # For each token, pick top-k experts
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)  # both [B, S, k]
        # Create a dispatch mask: 1 where expert is in top-k, else 0
        dispatch_mask = torch.zeros_like(scores)
        for k in range(self.top_k):
            dispatch_mask.scatter_(-1, topk_indices[..., k:k+1], topk_scores[..., k:k+1])
        # dispatch_mask: [B, S, num_experts] (can be soft if using topk_scores, or hard mask if set to 1)
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
        print(f"{next(self.experts[1].parameters()).device=}")
        self.expert_devices = list(expert_devices)  # Track current device for each expert
        print(self.expert_devices)

    def forward(self, x, routing_assignments):
        # x: [B, S, D] (main device)
        # routing_assignments: [B*S] int tensor, value is expert index for each token
        B, S, D = x.shape
        x_flat = x.view(B*S, D)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (routing_assignments == i)
            if mask.sum() == 0:
                continue

            expert_input = x_flat[mask].to(self.expert_devices[i])

            print(f"Expert {i} device:", next(expert.parameters()).device)
            print(f"Expert input device:", expert_input.device)
            print(f"Main x device:", x.device)
            # print(f"Expert output device:", expert_output.device)

            expert_output = expert(expert_input).to(x.device)
            out[mask] = expert_output
        return out.view(B, S, D)
