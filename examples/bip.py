"""Example for BatchIntrinsicPlasticity."""
import os
import numpy as np
from pyrcn.base.blocks import BatchIntrinsicPlasticity

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()


directory = os.path.join(os.getcwd(), 'bip')


def main():
    if not os.path.exists(directory):
        os.makedirs(directory)

    rs = np.random.RandomState(42)

    algorithm = 'dresden'
    sample_size = (1000, 1)

    i2n_uniform = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='uniform', algorithm=algorithm)
    i2n_exponential = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='exponential', algorithm=algorithm)
    i2n_normal = BatchIntrinsicPlasticity(
        hidden_layer_size=1, input_activation='tanh', random_state=rs,
        distribution='normal', algorithm=algorithm)

    X_uniform = rs.uniform(size=sample_size)
    X_exponential = rs.exponential(size=sample_size)
    X_normal = rs.normal(size=sample_size)

    def exponential(x, lam):
        return lam * np.exp(-lam * x)

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) \
            / np.sqrt(2. * np.pi) / sig

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
    fig, axs = plt.subplots(3, 4)
    # plt.ylabel('f_x')
    # plt.xlabel('f_y')
    # fig.suptitle('BIP transformations')
    bins = 20
    sns.histplot(data=i2n_exponential.fit_transform(X_exponential), bins=bins,
                 stat="density", ax=axs[0, 0], legend=False)
    axs[0, 0].set_xlim((-1., 1.))
    axs[0, 0].set_ylim((0., 3.))
    sns.histplot(data=i2n_normal.fit_transform(X_exponential), bins=bins,
                 stat="density", ax=axs[0, 1], legend=False)
    axs[0, 1].set_xlim((-1., 1.))
    axs[0, 1].set_ylim((0., 3.))
    sns.histplot(data=i2n_uniform.fit_transform(X_exponential), bins=bins,
                 stat="density", ax=axs[0, 2], legend=False)
    axs[0, 2].set_xlim((-1., 1.))
    axs[0, 2].set_ylim((0., 3.))

    sns.histplot(data=i2n_exponential.fit_transform(X_normal), bins=bins,
                 stat="density", ax=axs[1, 0], legend=False)
    axs[1, 0].set_xlim((-1., 1.))
    axs[1, 0].set_ylim((0., 1.5))
    sns.histplot(data=i2n_normal.fit_transform(X_normal), bins=bins,
                 stat="density", ax=axs[1, 1], legend=False)
    axs[1, 1].set_xlim((-1., 1.))
    axs[1, 1].set_ylim((0., 1.5))
    sns.histplot(data=i2n_uniform.fit_transform(X_normal), bins=bins,
                 stat="density", ax=axs[1, 2], legend=False)
    axs[1, 2].set_xlim((-1., 1.))
    axs[1, 2].set_ylim((0., 1.5))

    sns.histplot(data=i2n_exponential.fit_transform(X_uniform), bins=bins,
                 stat="density", ax=axs[2, 0], legend=False)
    axs[2, 0].set_xlim((-1., 1.))
    axs[2, 0].set_ylim((0., 2.5))
    sns.histplot(data=i2n_normal.fit_transform(X_uniform), bins=bins,
                 stat="density", ax=axs[2, 1], legend=False)
    axs[2, 1].set_xlim((-1., 1.))
    axs[2, 1].set_ylim((0., 2.5))
    sns.histplot(data=i2n_uniform.fit_transform(X_uniform), bins=bins,
                 stat="density", ax=axs[2, 2], legend=False)
    axs[2, 2].set_xlim((-1., 1.))
    axs[2, 2].set_ylim((0., 2.5))

    sns.histplot(data=X_exponential, bins=bins, ax=axs[0, 3], legend=False)
    axs[0, 3].set_title('exponential')
    sns.histplot(data=X_normal, bins=bins, ax=axs[1, 3], legend=False)
    axs[1, 3].set_title('normal')
    sns.histplot(data=X_uniform, bins=bins, ax=axs[2, 3], legend=False)
    axs[2, 3].set_title('uniform')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
