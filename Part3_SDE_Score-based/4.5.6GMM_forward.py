# Gaussion Mixture Model
# 要想构造一个混合高斯模型，首先要知道两个μ和convariance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal


def kdeplot(pnts, label='', ax=None, titlestr=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sns.kdeplot(x=pnts[:, 0], y=pnts[:, 1], ax=ax, label=label, **kwargs)
    if (titlestr is not None):
        ax.set_title(titlestr)


def quiver_plot(pnts, vecs, *args, **kwargs):
    plt.quiver(pnts[:, 0], pnts[:, 1], vecs[:, 0], vecs[:, 1], *args, **kwargs)


class GaussianMixture():
    def __init__(self, mus, covs, weights) -> None:
        self.n_components = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(cov) for cov in covs]  # precision matrices 求逆
        self.weights = weights
        self.norm_weights = weights / np.sum(weights)  # 归一化
        self.RVs = []  # Random Variable

        # 实例化一个二元高斯分布
        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))
        self.dims=len(mus[0])
    def sample(self, N):
        '''draw N samples from the mixture model'''
        rand_component = np.random.choice(self.n_components, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs])  # 现在两个分布各有5000个点
        gmm_samples = all_samples[rand_component, np.arange(N), :]
        return gmm_samples, rand_component, all_samples

    def score(self, x):
        # pdf probability density function概率密度函数
        # get the pdf of samples
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis:]
        # 参与度,其实是简单的归一化
        # 考虑到了点到两个分布中心的距离,,还考虑到了每个分布占有的权重
        participance = weighted_compon_pdf / np.sum(weighted_compon_pdf, axis=1, keepdims=True)

        scores = np.zeros_like(x)

        for i in range(self.n_components):
            gradvec = -(x - self.mus[i]) @ self.precs[i]  # 矩阵乘法
            scores += participance[:, i, np.newaxis] * gradvec  # (5000,2) 2表示的是feature dims.
        return scores

def marginal_prob_std_np(t, sigma):#这里需要标准差,需要开方std->now_sigma
    return np.sqrt(sigma**(2*t)-1)/(2*np.log(sigma))

def diffuse_gmm(gmm, dt, sigma):
    beta_t = marginal_prob_std_np(t, sigma) ** 2  # sigma->std(标准差),平方后得方差
    # 将这个lambda_t分别作用于两个Gaussion_distribution
    noise_cov = np.eye(gmm.dims) * beta_t#噪声矩阵
    covs_diff=[cov+noise_cov for cov in gmm.covs]#se them as two indepent gaussion disribution
    return GaussianMixture(gmm.mus,covs_diff,gmm.weights)

if __name__ == '__main__':
    # 2D Gaussion distribution  按照GMM.png的形状构造

    mu1 = np.array([0, 1.0])
    Conv1 = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

    mu2 = np.array([2.0, -1.0])
    Conv2 = np.array([[2.0, 0.5],
                      [0.5, 1.0]])

    # iterator solution
    gmm = GaussianMixture([mu1, mu2], [Conv1, Conv2], [1.0, 1.0])
    x0, _, _ = gmm.sample(2000)

    sigma = 5
    nsteps = 200

    x_traj = np.zeros((*x0.shape, nsteps,))
    x_traj[:, :, 0] = x0
    dt = 1 / nsteps
    for i in range(1, nsteps):
        t = i * dt
        eps_z = np.random.randn(*x0.shape)
        # central limit theorem 中心极限定理
        x_traj[:, :, i] = x_traj[:, :, i - 1] + eps_z * (sigma ** t) * np.sqrt(dt)

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 2, figsize=[12, 6])

    # Plot density of target distribution of x_0
    sns.kdeplot(x=x_traj[:, 0, 0], y=x_traj[:, 1, 0], ax=axs[0])

    axs[0].set_title("Density of Target distribution of $x_0$")
    axs[0].axis("equal")

    # analyzatical solution
    gmm_t = diffuse_gmm(gmm, nsteps / nsteps, sigma)
    samples_t, _, _ = gmm_t.sample(2000)  # 2000,2


    #plot density of x_T samples after nsteps step diffusion(both diffused and analytical GMM)
    sns.kdeplot(x=x_traj[:,0,nsteps-1],y=x_traj[:,1,nsteps-1],ax=axs[1],label='iter solution of GMM')
    sns.kdeplot(x=samples_t[:,0],y=samples_t[:,1],ax=axs[1],label='analytical solution of GMM')
    sns.kdeplot(x=x_traj[:, 0, -1], y=x_traj[:, 1, -1], ax=axs[1])
    axs[1].set_title(f'Density of $x_T$ samples after {nsteps} step diffusion')
    axs[1].axis('equal')
    axs[1].legend()
    plt.show()
    plt.savefig('target_dist_iterative_and_analytical_dist.png')