import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Exact transition probability for 1D diffusion
def transition_probability_diffusion_exact(x, t, params):
    x0, sigma = params['x0'], params['sigma']
    pdf = norm.pdf(x, loc=x0, scale=np.sqrt((sigma ** 2) * t))

    return pdf


def f_diff_sample(x, t, params):
    return np.zeros((*x.shape,))


def g_diff_sample(x, t, params):
    sigma = params['sigma']
    return sigma * np.ones((*x.shape,))


# 这里很类似模拟粉尘的运动，有决定项和不确定项，确定项是f_diff，不确定项是g_diff
def forward_SDE_simulation(x0, nsteps, f_diff, g_diff, params):
    t = 0
    x_trajectory = np.zeros((nsteps + 1, *x0.shape))
    x_trajectory[0] = np.copy(x0)
    # Perform many Euler-maryama time steps
    for i in range(nsteps):
        random_normal = np.random.randn(*x0.shape)
        x_trajectory[i + 1] = (
                x_trajectory[i] + f_diff(x_trajectory[i], t, params) *
                dt + g_diff(x_trajectory[i], t, params) * random_normal * np.sqrt(dt))
        t = t + dt
    return x_trajectory


if __name__ == '__main__':
    sigma = 1
    num_samples = 1000
    x0 = np.zeros(num_samples)  # 1000个粉尘粒子

    nsteps = 2000
    dt = 0.001  # 添加 dt 的定义
    T = nsteps * dt
    t = np.linspace(0, T, nsteps + 1)
    params = {'sigma': sigma, 'x0': x0, 'T': T}

    x_trajectory = forward_SDE_simulation(x0, nsteps, f_diff_sample, g_diff_sample, params)

    # 可视化
    plt.figure()  # 创建新的图像
    plt.hist(x_trajectory[0], bins=100)
    plt.title("$t = 0$", fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.savefig("diffusion_initial_distribution.png")  # 保存图片
    plt.show()  # 显示图片
    plt.clf()  # 清除当前图像

    x_f_min, x_f_max = np.amin(x_trajectory[-1]), np.amax(x_trajectory[-1])
    num_xf = 1000
    x_f_arg = np.linspace(x_f_min, x_f_max, num_xf)
    pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

    plt.figure()  # 创建新的图像
    plt.hist(x_trajectory[-1], bins=100)
    plt.plot(x_f_arg, pdf_final, color='black', linewidth=5)
    plt.title("$t = $" + str(T), fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.savefig("diffusion_final_distribution.png")  # 保存图片
    plt.show()  # 显示图片
    plt.clf()  # 清除当前图像

    # Plot some trajectories
    plt.figure()  # 创建新的图像
    sample_trajectories = [0, 1, 2, 3, 4]
    for s in sample_trajectories:
        plt.plot(t, x_trajectory[:, s])
    plt.title("Sample trajectories", fontsize=20)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("x", fontsize=20)
    plt.savefig("diffusion_trajectories.png")  # 保存图片
    plt.show()  # 显示图片
    plt.clf()  # 清除当前图像
