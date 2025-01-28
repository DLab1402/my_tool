import torch

# Create a tensor of shape (2, 3, 4)
tensor = torch.randn(2, 3, 4)

# Transpose to shape (2, 4, 3)
transposed_tensor = tensor.transpose(1, 2)

print(tensor)
print(transposed_tensor)  # Output: torch.Size([2, 4, 3])

