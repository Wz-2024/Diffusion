import torch
import os

device="gpu" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"]  ='1'
print(device)