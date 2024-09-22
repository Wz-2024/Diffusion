import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import torch


def kdeplot(pnts, label="", ax=None, titlestr=None, **kwargs):
    if ax is None:
        ax = plt.gca()  # figh, axs = plt.subplots(1,1,figsize=[6.5, 6])
    sns.kdeplot(x=pnts[:, 0], y=pnts[:, 1], ax=ax, label=label, **kwargs)
    if titlestr is not None:
        ax.set_title(titlestr)


def quiver_plot(pnts, vecs, *args, **kwargs):
    plt.quiver(pnts[:, 0], pnts[:, 1], vecs[:, 0], vecs[:, 1], *args, **kwargs)

def sample_X_and_score(gmm, trainN=10000, testN=2000):
    X_train_,_=gmm.sample(trainN)
    y_train=gmm.score(X_train)#analitical score
    X_test,_,_=gmm.sample(testN)
    y_test=gmm.score(X_test)
    #将他们变成tensor
    X_train_tsr=torch.tensor(X_train).float()
    y_train_tsr=torch.tensor(X_test).float()
    X_test_tsr=torch.tensor(X_test).float()
    y_test_tsr=torch.tensor(y_test).float()
    return X_train_tsr,y_train_tsr,X_test_tsr,y_test_tsr




def sample_X_and_score_t_depend(gmm, sigma=5, trainN=10000, testN=2000, partition=100,
                                EPS=0.02):
    trainN_part,testN_part=trainN//partition,testN//partition
    X_train_list,y_train_list,X_test_list,y_test_list,T_train_list,T_test_list=[],[],[],[],[],[]
    for t in np.linespace(0, 1, partition):#0~T->0~1
        gmm_t=diffuse_gmm(gmm,t,sigma)#analytical solution ti diffuse the GMM
        #对于给定的GMM,去采样
        #训练集-----标签          测试集------标签
        X_train_tsr,y_train_tsr,X_test_tsr,y_test_tsr=sample_X_and_score(gmm_t,trainN=trainN_part,testN=testN_part)
        T_train_tsr,T_test_tsr=torch.ones_like(y_train_tsr)*t,torch.ones_like(y_test_tsr)*t

        X_train_list.append(X_train_tsr)
        y_train_list.append(y_train_tsr)
        X_train_list.append(T_train_tsr)
        X_test_list.append(X_test_tsr)
    X_train_tsr = torch.cat(X_train_list, dim=0)
    y_train_tsr = torch.cat(y_train_list, dim=0)
    X_test_tsr = torch.cat(X_test_list, dim=0)
    y_test_tsr = torch.cat(y_test_list, dim=0)
    T_train_tsr = torch.cat(T_train_list, dim=0)
    T_test_tsr = torch.cat(T_test_list, dim=0)
    return X_train_tsr, y_train_tsr, T_train_tsr, X_test_tsr, y_test_tsr, T_test_tsr


class GaussianMixture():
    def __init__(self, mus, covs, weights) -> None:
        self.n_components = len(mus)
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(cov) for cov in covs]  # precision matrices
        self.weights = weights  # e.g. 1:1
        self.norm_weights = weights / np.sum(weights)  # 0.5:0.5
        self.RVs = []

        for i in range(len(mus)):
            self.RVs.append(multivariate_normal(mus[i], covs[i]))

        self.dims = len(mus[0])

    def score(self, x):  # x (5000, 2)
        component_pdf = np.array([rv.pdf(x) for rv in self.RVs]).T
        weighted_compon_pdf = component_pdf * self.norm_weights[np.newaxis, :]
        participance = weighted_compon_pdf / np.sum(weighted_compon_pdf, axis=1, keepdims=True)
        # (1) distance to mus, (2) weight assigned to each component

        scores = np.zeros_like(x)  # (5000, 2) ！！！！

        for i in range(self.n_components):
            gradvec = -(x - self.mus[i]) @ self.precs[i]
            scores += participance[:, i, np.newaxis] * gradvec  # (5000, 2) ！2->feaeture dims instead of 2 compoents

        return scores

    def sample(self, N):
        """draw N samples from the mixture model"""
        rand_component = np.random.choice(self.n_components, size=N, p=self.norm_weights)
        all_samples = np.array([rv.rvs(N) for rv in self.RVs])  # 5000 dog samples + 5000 wolf samples
        gmm_samples = all_samples[rand_component, np.arange(N),
                      :]  # 2, 5000, 2: 2 components, 5000 samples, 2D Gaussian distribution
        return gmm_samples, rand_component, all_samples


def marginal_prob_std_np(t, sigma):  # std->new_sigma
    return np.sqrt(
        (sigma ** (2 * t) - 1) / (2 * np.log(sigma)))  # return a std(i.e. sigma)instead of a var(i.e. sigma**2)


def diffuse_gmm(gmm, t, sigma):
    beta_t = marginal_prob_std_np(t, sigma) ** 2  # sigma --> std   sigma**2 --> var
    noise_cov = np.eye(gmm.dims) * beta_t
    covs_diff = [cov + noise_cov for cov in gmm.covs]  # see them as two indepent gaussian distriubtion
    return GaussianMixture(gmm.mus, covs_diff, gmm.weights)


if __name__ == "__main__":
    mu1 = np.array([0, 1.0])  # 2D Gaussian distribution
    Cov1 = np.array([[1.0, 0.0],
                     [0.0, 1.0]])  # covariance matrix 2x2

    mu2 = np.array([2.0, -1.0])
    Cov2 = np.array([[2.0, 0.5],
                     [0.5, 1.0]])

    ##### iterative solution
    gmm = GaussianMixture([mu1, mu2], [Cov1, Cov2], [1.0, 1.0])

    nsteps=1000
    Train_sampleN=100000
    Test_sampleN=2000
    sigma=10


    X_train, y_train,T_train, X_test, y_test,T_test=sample_X_and_score_t_depend(
        gmm,
        sigma=sigma,
        trainN=100000,
        testN=2000,
        partition=100,
        EPS=0.0001
    )


############
def reverse_diffusion_time_dep(score_model_td, sampN=500, sigma=5, nsteps=200, ndim=2, exact=False):
    betaT = (sigma ** 2 - 1) / (2 * np.log(sigma))
    xT = np.sqrt(betaT) * np.random.randn(sampN, ndim)
    x_traj_rev = np.zeros((*xT.shape, nsteps,))
    x_traj_rev[:, :, 0] = xT
    dt = 1 / nsteps
    for i in range(1, nsteps):
        t = 1 - i * dt
        tvec = torch.ones((sampN)) * t
        eps_z = np.random.randn(*xT.shape)
        if exact:  # given a known GMM
            gmm_t = diffuse_gmm(score_model_td, t, sigma)  # score_model_td is gmm
            score_xt = gmm_t.score(x_traj_rev[:, :, i - 1])  # <====== (1) given known gmm_t, gmm_t.score()
        else:  # the target distribution is unknown, use a model to learn the score. Check next section
            with torch.no_grad():
                # score_xt = score_model_td(torch.cat((torch.tensor(x_traj_rev[:,:,i-1]).float(),tvec),dim=1)).numpy()
                score_xt = score_model_td(torch.tensor(x_traj_rev[:, :, i - 1]).float(),
                                          tvec).numpy()  # (2)<===== score = model.predict(x, t)
        x_traj_rev[:, :, i] = x_traj_rev[:, :, i - 1] + sigma ** (2 * t) * score_xt * dt + (sigma ** t) * np.sqrt(
            dt) * eps_z

    return x_traj_rev