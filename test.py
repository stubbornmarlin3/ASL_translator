from asl import ASL
from i3d import I3D
import torch

model = ASL(10)

print(sum(p.numel() for p in model.parameters()))