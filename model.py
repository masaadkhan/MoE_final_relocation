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

    def forward(self, x, k=1):
        logits = self.gate(x)
        topk = torch.topk(logits, k, dim=-1)
        routes = topk.indices
        return routes

class ExpertParallelMoE(nn.Module):
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model).to(f'cuda:{i}') for i in range(num_experts)])
        self.num_experts = num_experts

    def forward(self, x, k=1, stats=None):
        batch, seq, d_model = x.size()
        device = x.device

        routes = self.router(x, k=k)
        routes_flat = routes.reshape(-1, k)
        x_flat = x.reshape(-1, d_model)

        if stats is not None:
            for i in range(self.num_experts):
                stats[i] += (routes_flat == i).sum().item()

        expert_inputs = [[] for _ in range(self.num_experts)]
        for idx, experts in enumerate(routes_flat):
            for expert_id in experts.tolist():
                expert_inputs[expert_id].append(x_flat[idx].detach().cpu())

        expert_inputs_tensor = []
        for i, expert_batch in enumerate(expert_inputs):
            if expert_batch:
                #TODO(MASAAD): What does this do?...
                batch_tensor = torch.stack(expert_batch).to(f'cuda:{i}')
            else:
                batch_tensor = torch.zeros((0, d_model), device=f'cuda:{i}')
            expert_inputs_tensor.append(batch_tensor)

        expert_outputs = []
        #TODO(MASAAD): What does this do?...
        # Why are we zipping this?...
        for i, (expert, input_tensor) in enumerate(zip(self.experts, expert_inputs_tensor)):
            if input_tensor.shape[0] == 0:
                out = torch.zeros_like(input_tensor)
            else:
                out = expert(input_tensor)
            expert_outputs.append(out)

        #TODO(MASAAD): What does this do?...
        outputs = torch.cat([out.detach().cpu() for out in expert_outputs], dim=0).to(device)
        outputs = outputs[:batch*seq]
        outputs = outputs.reshape(batch, seq, d_model)
        return outputs
