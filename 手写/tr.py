import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(device)

from datasets import tqdm
print(3)