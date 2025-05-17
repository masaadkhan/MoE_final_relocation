import torch
import numpy as np
from model import ExpertParallelMoE, Expert

def migrate_expert(moe, expert_idx, src_gpu, dst_gpu):
    src_expert = moe.experts[expert_idx].to(f'cuda:{src_gpu}')
    expert_state = {k: v.cpu() for k, v in src_expert.state_dict().items()}
    moe.experts[expert_idx] = Expert(src_expert.ff[0].in_features).to(f'cuda:{dst_gpu}')
    moe.experts[expert_idx].load_state_dict({k: v.to(f'cuda:{dst_gpu}') for k, v in expert_state.items()})
    print(f"Expert {expert_idx} migrated from GPU {src_gpu} to GPU {dst_gpu}.")

def train_moe(num_epochs=5, k=2):
    world_size = torch.cuda.device_count()
    d_model = 512
    batch_size = 32
    seq_len = 32

    moe = ExpertParallelMoE(d_model, world_size)
    stats = np.zeros(world_size, dtype=int)

    for epoch in range(num_epochs):
        stats[:] = 0
        x = torch.randn(batch_size, seq_len, d_model).to('cuda:0')
        outputs = moe(x, k=k, stats=stats)
        print(f"Epoch {epoch} token distribution:", stats)
        if epoch < num_epochs - 1:
            migrate_expert(moe, 0, src_gpu=epoch % world_size, dst_gpu=(epoch + 1) % world_size)

if __name__ == "__main__":
    train_moe(num_epochs=3, k=2)
