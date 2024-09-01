import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

pipeline = DiffusionPipeline.from_pretrained("/data_disk/dyy/models").to(device)
#查看/data_disk/dyy/models是否存在
image=pipeline(prompt="a and nylon stockings,European girl", num_inference_steps=50).images[0]
print(type(image))
#将这片图片显示出来
plt.imshow(image)
plt.axis('off')
plt.show()

print(pipeline)