import torch

import os
import transformers
# from torch.utils.tensorboard.summary import image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
