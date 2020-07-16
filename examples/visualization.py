import numpy as np
from echo_state_network import ESNRegressor
import matplotlib.pylab as plt
# Latex Backend for Matplotlib
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'serif',
    'text.usetex': True,
    'figure.figsize': [12, 8],
   }

base_esn = ESNRegressor(k_in=10, input_scaling=1.0, spectral_radius=0.9, bias=0.0, ext_bias=False, leakage=0.1,
                        reservoir_size=50, k_res=10, reservoir_activation='tanh', bi_directional=False,
                        teacher_scaling=1.0, teacher_shift=0.0, solver='ridge', beta=1e-6, random_state=10)
X = np.zeros(shape=(100, 10), dtype=int)
X[5] = 1
y = X[:, 0]

base_esn.fit(X=X, y=y, n_jobs=0)
y_pred = base_esn.predict(X=X, keep_reservoir_state=True)

fig, ax = plt.subplots()
im = ax.imshow(base_esn.reservoir_state[:, 1:].T)
ax.set_xlim([0, base_esn.reservoir_state.shape[0]])
ax.set_ylim([0, base_esn.reservoir_state.shape[1] - 1])
ax.set_xlabel(r'k')
ax.set_ylabel(r'X[k]')
plt.grid()
plt.savefig("is_10_sr_09_lr_01.pdf")
exit(0)
