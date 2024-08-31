import torch
print(torch.__version__)
import matplotlib.pyplot as plt
print(3)

if torch.cuda.is_available():
    print('yes')
else :
    print('no')


import matplotlib.pyplot as plt
import numpy as np

# 创建一些数据
x = np.linspace(0, 5, 100)
y = np.sin(x)

# 创建图像
plt.plot(x, y)
plt.title('Sine Wave')

# 保存图像到当前目录
plt.savefig('./UNet-based_diffusion.png')

# 显示图像（可选）
plt.show()
