import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Exact transition probability for 1D diffusion
def transition_probability_diffusion_exact(x, t, params):
  x0, sigma= params['x0'], params['sigma']

  pdf = norm.pdf(x, loc=x0, scale=np.sqrt((sigma**2)*t))  # pdf of normal distribution with mean x0 and variance (sigma^2)*t
  return pdf




# Drift function
def f_diff_sample(x, t, params):
    # 当前模拟无外力的状态,
    return np.zeros((*x.shape,))


def g_diff_sample(x, t, params):
    sigma = params['sigma']
    return sigma * np.ones((*x.shape,))


# Simulate SED with drift function f and noise amplitude g for arbitrary number of steps
def forward_SDE_simulation(x0, nsteps,dt, f, g, params):
    t = 0
    x_traj = np.zeros((nsteps + 1, *x0.shape))
    x_traj[0] = np.copy(x0)
    # Perform the Euler-Maruyama time steps
    for i in range(nsteps):
        random_normal = np.random.randn(*x0.shape)

        # 这里模拟迭代过程 x_traj[i+1]就是等式左边的x(t+dt)
        x_traj[i + 1] = x_traj[i] + f(x_traj[i], t,params) * dt + g(x_traj[i], t, params) * np.sqrt(dt) * random_normal
        t = t + dt
    return x_traj


if __name__ == "__main__":
    # 这里用离散的写法
    sigma = 1
    num_samples = 1000
    x0 = np.zeros(num_samples)  # 模拟1000个粉尘例子
    nsteps = 2000
    dt = 0.001

    T = nsteps * dt
    t = np.linspace(0, T, nsteps + 1)

    params = {'sigma': sigma, "x0": x0, "T": T}

    # f_diff是决定项,,,g_diff是随机项(noise项)
    x_traj = forward_SDE_simulation(x0, nsteps, dt, f_diff_sample, g_diff_sample, params)


    #可视化
    plt.hist(x_traj[0], bins=100)
    plt.title("$t = 0$", fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.show()
    plt.savefig("diffusion_initial_distribution.png")
    plt.clf()

    x_f_min, x_f_max = np.amin(x_traj[-1]), np.amax(x_traj[-1])
    num_xf = 1000
    x_f_arg = np.linspace(x_f_min, x_f_max, num_xf)
    pdf_final = transition_probability_diffusion_exact(x_f_arg, T, params)

    plt.hist(x_traj[-1], bins=100,density=True)
    plt.plot(x_f_arg, pdf_final, color='black', linewidth=5)
    plt.title("$t = $" + str(T), fontsize=20)
    plt.xlabel("$x$", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.show()
    plt.savefig("diffusion_final_distribution.png")
    plt.clf()

    # Plot some trajectories
    sample_trajectories = [0, 1, 2, 3, 4]
    for s in sample_trajectories:
        plt.plot(t, x_traj[:, s])
    plt.title("Sample trajectories", fontsize=20)
    plt.xlabel("$t$", fontsize=20)
    plt.ylabel("x", fontsize=20)
    plt.show()
    plt.savefig("diffusion_trajectories.png")