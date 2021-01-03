import torch

a = torch.Tensor(2, 3)
print(a)
# tensor([[0.0000, 0.0000, 0.0000],
#        [0.0000, 0.0000, 0.0000]])

print(a.view(1, -1))
# tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])