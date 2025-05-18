import torch
import torch.nn as nn

class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2, noise_std=0.5):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.noise_std = noise_std

    def forward(self, x):
        B, S, D = x.shape
        logits = self.gate(x)

        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        scores = torch.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        dispatch_mask = torch.zeros_like(scores)

        for k in range(self.top_k):
            dispatch_mask.scatter_(-1, topk_indices[..., k:k+1], topk_scores[..., k:k+1])
        return dispatch_mask

class Expert(nn.Module):
    def __init__(self, d_model, expert_id):
        super().__init__()
        self.id = expert_id
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
            Expert(d_model, i).to(expert_devices[i]) for i in range(len(expert_devices))
        ])
        self.expert_devices = list(expert_devices)  # Track current device for each expert

    def forward(self, x, return_stats=False):
        B, S, D = x.shape
        dispatch_mask = self.router(x)
        x_flat = x.view(B*S, D)
        out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            device_i = self.expert_devices[i]

            weights = dispatch_mask[..., i].reshape(-1)           # (B·S,)
            mask    = weights > 0

            # --- stats (optional) ---------------------------------------------------
            tokens_per_expert = (dispatch_mask > 0).sum(dim=(0, 1))
            # ------------------------------------------------------------------------

            if not mask.any():
                continue                                          # nothing for this expert

            # Move *both* tensors to the expert’s GPU *before* math
            expert_input = x_flat[mask].to(device_i, non_blocking=True)
            weights_i    = weights[mask].to(device_i, non_blocking=True)

            weighted_input = expert_input * weights_i.unsqueeze(1)
            expert_output  = expert(weighted_input)               # runs on device_i

            # bring result back to the original device of `x`
            out[mask] = expert_output.to(x.device, non_blocking=True)

        if return_stats:
            return out.view(B, S, D), tokens_per_expert
        else:
            return out.view(B, S, D)
    
    def expert_to_gpu(self, expert_id):
        return next(self.experts[expert_id].parameters()).device

    def swap_experts(self, A, B):
        device_a = self.expert_to_gpu(A)
        device_b = self.expert_to_gpu(B)

        tmp_expert = self.experts[A]
        self.experts[A] = self.experts[B].to(device_b)
        self.experts[B] = tmp_expert.to(device_a)

        self.expert_devices[A] = device_b
        self.expert_devices[B] = device_a

        print(f"Swapped expert {A} (now on {device_b}) with expert {B} (now on {device_a})")
