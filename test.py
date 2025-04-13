import torch
from asl import ASL

model = ASL()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)