import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self): super().__init__(); self.ff = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 512))
    def forward(self, x): print("Expert got x.requires_grad =", x.requires_grad); return self.ff(x)

x = torch.randn(8, 512, device='cuda')
e = Expert().cuda()
mask = torch.ones_like(x)  # or mask of shape [8,512] if you want elementwise

y = e(x) * mask
print("y.requires_grad =", y.requires_grad)
loss = y.sum()
print("loss.requires_grad =", loss.requires_grad)
loss.backward()
