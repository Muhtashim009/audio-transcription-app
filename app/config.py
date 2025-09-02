import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
