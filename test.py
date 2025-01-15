import torch

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)

print(input)
print(target)