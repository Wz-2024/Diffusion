import numpy as np
import matplotlib.pyplot as plt


def noise_strength_constant(t):
    return 1

def forward_diffusion_1D(x0,t,dt,nsteps,noise_strength_fn):
    # Initialize trajectory初始化一个轨迹
    x=np.zeros((nsteps+1,x0.shape[0]),dtype=float)
    x[0]=x0
    #ts表示所有的t
    ts=[t]#Initialize time array wth starting time
    #计算从t到t+dt的轨迹
    for i in range(nsteps):
        noise_strength=noise_strength_fn(t)
        random_normal=np.random.randn(x0.shape[0])
        x[i+1]=x[i]+np.sqrt(dt)*noise_strength*random_normal #这里的random_normal是正态分布的随机数,r
        t=t+dt
        ts.append(t)
    return x,ts#x是矩阵,ts是数组用来画图



if __name__ == '__main__':
    nsteps=100
    t=0
    dt=0.1
    # 这里相当于将sigma的值固定为一个常数
    noise_strength_fn=noise_strength_constant

    #五个粒子
    num_particles=5
    x0=np.zeros((num_particles))
    x,ts=forward_diffusion_1D(x0,t,dt,nsteps,noise_strength_fn)

    plt.figure()
    plt.plot(ts,x)
    plt.xlabel('time',fontsize=20)
    plt.ylabel('$x$',fontsize=20)
    plt.title('Forward Diffusion Visualized',fontsize=20)
    plt.savefig('1d-forward-diffusion(2).png')

    plt.c


