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
            scores += participance[:, i, np.newaxis] * gradvec#(5000,2) 2表示的是feature dims.
        return scores

if __name__ == '__main__':
    # 2D Gaussion distribution  按照GMM.png的形状构造

    mu1 = np.array([0, 1.0])
    Conv1 = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

    mu2 = np.array([2.0, -1.0])
    Conv2 = np.array([[2.0, 0.5],
                      [0.5, 1.0]])

    # 两个Gaussion distribution 和 权重
    gmm = GaussianMixture([mu1, mu2], [Conv1, Conv2], [1.0, 1.0])

    # 假设gmm_samps是5000个样本  rand_component取值为0或1,表示取的点服从哪个高斯分布  component_samples计数
    gmm_samples, rand_component, all_samples = gmm.sample(5000)

    scorevecs = gmm.score(gmm_samples)

    print('gmm_samples', gmm_samples.shape)
    print('rand_component', rand_component.shape)
    print('all_samples', all_samples.shape)

    # output
    # >>> gmm_samples  (5000, 2) # 5000 samples, 2D Gaussian distribution
    # rand_component  (5000,) # 0, 1
    # all_samples  (2, 5000, 2) -->  (0, 3, 1)  0表示第0个高斯分布,3表示第三个点,2表示得到的'脚身比'
    # scorevecs  (5000, 2) # 2 --> feature dims.     ---> 2 components


    figh, ax = plt.subplots(1,1,figsize=[6,6])
    kdeplot(all_samples[0,:,:], label="comp1", )
    kdeplot(all_samples[1,:,:], label="comp2", )
    plt.title("Empirical_density_of_each_component")
    plt.legend()
    plt.axis("image")
    plt.savefig("Empirical_density_of_each_component.png")
    plt.clf()

    figh, ax = plt.subplots(1,1,figsize=[6,6])
    kdeplot(gmm_samples, )
    plt.title("Empirical density of Gaussian mixture density")
    plt.axis("image")
    plt.savefig("Empirical_density_of_Gaussian_mixture_density.png")
    plt.clf()

    plt.figure(figsize=[8,8])
    quiver_plot(gmm_samples, scorevecs)
    plt.title("Score vector field $\log p(x)$")
    plt.axis("image")
    plt.savefig("Score_vector_field_log_p(x).png")
    plt.clf()