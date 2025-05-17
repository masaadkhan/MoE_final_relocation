import torch
import torch.nn as nn

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
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts

    def forward(self, x, top_k=1):
        assert top_k <= self.num_experts, f"{top_k=} can't exceed number of experts={self.num_experts}"
        logits = self.gate(x)
        topk = torch.topk(logits, top_k, dim=-1)
        routes = topk.indices
        return routes

class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, num_experts, expert_capacity):
        super().__init__()
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model).to(f'cuda:{i}') for i in range(num_experts)])
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

    def forward(self, x, top_k=1, stats=None):
        batch, seq, d_model = x.size()
        device = x.device

        routes = self.router(x, top_k=top_k)
        x_flat = x.reshape(-1, d_model)
        routes_flat = routes.reshape(-1, top_k)
        indices_flat = torch.arange(x_flat.size(0), device=x.device).unsqueeze(-1).expand(-1, top_k) # [batch*seq, k]

        if stats is not None:
            for i in range(self.num_experts):
                stats[i] += (routes_flat == i).sum().item()
        
        expert_token_indices = [
            indices_flat[routes_flat == i][:self.expert_capacity]  # [≤ expert_capacity]
            for i in range(self.num_experts)
        ]

        # Pad indices if needed so each is exactly expert_capacity long
        expert_token_indices_padded = []
        for idxs in expert_token_indices:
            if idxs.numel() < self.expert_capacity:
                # Pad with -1 (will be ignored later)
                pad = torch.full((self.expert_capacity - idxs.numel(),), -1, device=device, dtype=torch.long)
                idxs = torch.cat([idxs, pad], dim=0)
            expert_token_indices_padded.append(idxs)  # [expert_capacity]
        
        # Build expert input tensors
        expert_inputs_tensor = []
        for i, idxs in enumerate(expert_token_indices_padded):
            # Gather tokens for this expert (ignore -1s)
            valid_mask = (idxs != -1)
            num_valid = valid_mask.sum().item()
            if num_valid > 0:
                batch_tensor = torch.zeros((self.expert_capacity, d_model), device=f'cuda:{i}')
                batch_tensor[:num_valid] = x_flat[idxs[valid_mask]].to(f'cuda:{i}')
            else:
                batch_tensor = torch.zeros((self.expert_capacity, d_model), device=f'cuda:{i}')
            expert_inputs_tensor.append(batch_tensor)

        # Forward through experts
        expert_outputs = []
        for i, (expert, input_tensor) in enumerate(zip(self.experts, expert_inputs_tensor)):
            out = expert(input_tensor)  # [expert_capacity, d_model]
            expert_outputs.append(out)

        # Gather outputs back, scatter to correct positions
        # Make an output buffer for all tokens
        output_flat = torch.zeros(x_flat.size(0), d_model, device=device)
        for i, idxs in enumerate(expert_token_indices_padded):
            valid_mask = (idxs != -1)
            if valid_mask.sum() > 0:
                output_flat[idxs[valid_mask]] = expert_outputs[i][valid_mask].to(device)

        outputs = output_flat.reshape(batch, seq, d_model)
        return outputs
