"""
Example for BatchIntrinsicPlasticity
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

from pyrcn.base import InputToNode, BatchIntrinsicPlasticity


def main():
    rs = np.random.RandomState(42)

    algorithm = 'dresden'
    sample_size = (1000, 1)

    i2n_uniform = BatchIntrinsicPlasticity(hidden_layer_size=1, activation='tanh', random_state=rs, distribution='uniform', algorithm=algorithm)
    i2n_exponential = BatchIntrinsicPlasticity(hidden_layer_size=1, activation='tanh', random_state=rs, distribution='exponential', algorithm=algorithm)
    i2n_normal = BatchIntrinsicPlasticity(hidden_layer_size=1, activation='tanh', random_state=rs, distribution='normal', algorithm=algorithm)

    X_uniform = rs.uniform(size=sample_size)
    X_exponential = rs.exponential(size=sample_size)
    X_normal = rs.normal(size=sample_size)

    def exponential(x, lam):
        return lam * np.exp(-lam * x)

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2. * np.pi) / sig

    # X_uniform = np.linspace(start=-1., stop=1., num=1000).reshape(-1, 1)
    # X_exponential = exponential(X_uniform + 1., 1)
    # X_normal = gaussian(X_uniform, 0, 1)

    """
        y_uni_exp = i2n_exponential.fit_transform(X_uniform)
        y_exp_exp = i2n_exponential.fit_transform(X_exponential)
        y_norm_exp = i2n_exponential.fit_transform(X_normal)
    
        y_uni_uni = i2n_uniform.fit_transform(X_uniform)
        y_exp_uni = i2n_uniform.fit_transform(X_exponential)
        y_norm_uni = i2n_uniform.fit_transform(X_normal)
    
        y_uni_norm = i2n_normal.fit_transform(X_uniform)
        y_exp_norm = i2n_normal.fit_transform(X_exponential)
        y_norm_norm = i2n_normal.fit_transform(X_normal)
    """

    # display distributions
    fig, axs = plt.subplots(3, 4, figsize=(8, 6))
    # plt.ylabel('f_x')
    # plt.xlabel('f_y')
    fig.suptitle('BIP transformations')

    bins = 20

    axs[0, 0].hist(i2n_exponential.fit_transform(X_exponential), bins=bins, density=True, color='b')
    axs[0, 0].set_xlim((-1., 1.))
    axs[0, 1].hist(i2n_normal.fit_transform(X_exponential), bins=bins, density=True, color='g')
    axs[0, 1].set_xlim((-1., 1.))
    axs[0, 2].hist(i2n_uniform.fit_transform(X_exponential), bins=bins, density=True, color='r')
    axs[0, 2].set_xlim((-1., 1.))

    axs[1, 0].hist(i2n_exponential.fit_transform(X_normal), bins=bins, density=True, color='b')
    axs[1, 0].set_xlim((-1., 1.))
    axs[1, 1].hist(i2n_normal.fit_transform(X_normal), bins=bins, density=True, color='g')
    axs[1, 1].set_xlim((-1., 1.))
    axs[1, 2].hist(i2n_uniform.fit_transform(X_normal), bins=bins, density=True, color='r')
    axs[1, 2].set_xlim((-1., 1.))

    axs[2, 0].hist(i2n_exponential.fit_transform(X_uniform), bins=bins, density=True, color='b')
    axs[2, 0].set_xlim((-1., 1.))
    axs[2, 1].hist(i2n_normal.fit_transform(X_uniform), bins=bins, density=True, color='g')
    axs[2, 1].set_xlim((-1., 1.))
    axs[2, 2].hist(i2n_uniform.fit_transform(X_uniform), bins=bins, density=True, color='r')
    axs[2, 2].set_xlim((-1., 1.))

    axs[0, 3].hist(X_exponential, bins=bins, color='gray')
    axs[0, 3].set_title('exponential')
    axs[1, 3].hist(X_normal, bins=bins, color='gray')
    axs[1, 3].set_title('normal')
    axs[2, 3].hist(X_uniform, bins=bins, color='gray')
    axs[2, 3].set_title('uniform')

    plt.tight_layout()
    plt.savefig("bip-transformations.pdf")
    # plt.show()


if __name__ == "__main__":
    main()

"""
statistic, pvalue = scipy.stats.ks_1samp(y_test, scipy.stats.uniform.cdf)
assert statistic < pvalue
print("Kolmogorov-Smirnov does not reject H_0: y is uniformly distributed in [-.75, .75]")
"""
