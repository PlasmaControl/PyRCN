import glob
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid, cross_validate

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from joblib import Parallel, delayed, dump, load
from pyrcn.echo_state_network import SeqToLabelESNClassifier
from pyrcn.base import PredefinedWeightsInputToNode, NodeToNode
from pyrcn.metrics import accuracy_score, classification_report, confusion_matrix
from pyrcn.model_selection import SequentialSearchCV
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
#Options
plt.rc('image', cmap='RdBu')
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import librosa
import librosa.display


all_files = glob.glob(r"E:\free-spoken-digit-dataset\recordings\*.wav")
print(len(all_files))

X_train = []
X_test = []
y_train = []
y_test = []
print("extracting features...")
for k, f in enumerate(all_files):
    basename = os.path.basename(f).split('.')[0]
    # Get label (0-9) of recording.
    label = int(basename.split('_')[0])
    idx = int(basename.split('_')[2])
    # Load the audio signal and normalize it.
    x, sr = librosa.core.load(f, sr=None, mono=False)
    # x /= np.max(np.abs(x))
    mfcc = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(0.01*sr), n_fft=256, htk=True, n_mels=100, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    if idx <= 4:
        X_test.append(mfcc.T)
        y_test.append(label)
    else:
        X_train.append(mfcc.T)
        y_train.append(label)
print("done!")

scaler = StandardScaler().fit(X=np.vstack(X_train + X_test))
X_train_scaled = np.empty(shape=(len(X_train),), dtype=object)
X_test_scaled = np.empty(shape=(len(X_test),), dtype=object)
y_train = np.array(y_train, dtype=object)
X_train, X_train_scaled, y_train = shuffle(X_train, X_train_scaled, y_train)
y_test = np.array(y_test, dtype=object)
for k in range(len(X_train)):
    X_train_scaled[k] = scaler.transform(X_train[k])
    y_train[k] = np.atleast_1d(y_train[k]).astype(int)
for k in range(len(X_test)):
    X_test_scaled[k] = scaler.transform(X_test[k])
    y_test[k] = np.atleast_1d(y_test[k]).astype(int)

print(len(X_train), len(y_train), X_train[0].shape, y_train[0])
print(len(X_test), len(y_test), X_test[0].shape, y_test[0])

"""
initially_fixed_params = {'hidden_layer_size': 400,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bidirectional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = SeqToLabelESNClassifier(**initially_fixed_params)

try:
    sequential_search = load("../sequential_search_fsdd.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train_scaled, y_train)
    dump(sequential_search, "../sequential_search_fsdd.joblib")

esn = sequential_search.best_estimator_
y_pred_train = esn.predict(X=X_train_scaled)
cm = confusion_matrix(y_train, y_pred_train)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
plt.show()
print("Classification training report for estimator %s:\n%s\n" % (esn, classification_report(y_train, y_pred_train, digits=10)))

y_pred_test = esn.predict(X=X_test_scaled)
cm = confusion_matrix(y_test, y_pred_test)
cm_display = ConfusionMatrixDisplay(cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).plot()
plt.show()
print("Classification test report for estimator %s:\n%s\n" % (esn, classification_report(y_test, y_pred_test, digits=10)))

df = pd.DataFrame(sequential_search.all_cv_results_["step1"])
df.mean_test_score = df.mean_test_score.clip(lower=0.95)
fig = plt.figure()
ax = sns.scatterplot(x="param_spectral_radius", y="param_input_scaling", hue="mean_test_score", palette='RdBu', data=df)
plt.xlabel("Spectral Radius")
plt.ylabel("Input Scaling")

norm = plt.Normalize(0.95, 1.0)
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
sm.set_array([])
plt.xlim((0, 2.01))
plt.ylim((0, 1.03))

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)
fig.set_size_inches(2, 1.25)
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.grid()
plt.savefig('Basic_ESN_IS_SR.pdf', bbox_inches='tight', pad_inches=0)

df = pd.DataFrame(sequential_search.all_cv_results_["step2"])
fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.lineplot(data=df, x="param_leakage", y="mean_test_score")
ax.set_xscale('log')
plt.xlabel("Leakage")
plt.ylabel("Score")
plt.xlim((1e-5, 1e0))
plt.ylim((0.8, 1.0))
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.grid()
plt.savefig('Basic_ESN_LR.pdf', bbox_inches='tight', pad_inches=0)

df = pd.DataFrame(sequential_search.all_cv_results_["step3"])
fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.lineplot(data=df, x="param_bias_scaling", y="mean_test_score")
plt.xlabel("Bias Scaling")
plt.ylabel("Score")
plt.xlim((0, 1))
plt.ylim((0.96, 0.98))
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.grid()
plt.savefig('Basic_ESN_BS.pdf', bbox_inches='tight', pad_inches=0)

idx = np.random.randint(0, 400, 50)

fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.heatmap(data=sequential_search.best_estimator_.input_to_node.input_weights.todense()[:, idx], cmap="RdBu", vmin=-1.0, vmax=1.0)
ax.invert_yaxis()
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
plt.xlabel("Neuron Index")
plt.ylabel("Input Dimension")
# plt.imshow(sequential_search.best_estimator_.input_to_node.input_weights.todense()[:, idx])
# plt.colorbar()
plt.savefig('Basic_ESN_Input_Weights.pdf', bbox_inches='tight', pad_inches=0)

print(sequential_search.all_best_params_)
print(sequential_search.all_best_score_)
base_esn = clone(sequential_search.best_estimator_)

param_grid = {'hidden_layer_size': [50, 100, 200, 400, 800, 1600, 3200, 6400],
              'random_state': range(1, 11)}

for params in ParameterGrid(param_grid):
    esn = clone(base_esn).set_params(**params).fit(X=X_train_scaled, y=y_train, n_jobs=8)
    score = accuracy_score(y_test, esn.predict(X_test_scaled))
    print("ESN with params {0} achieved score of {1}".format(params, score))


"""
kmeans = MiniBatchKMeans(n_clusters=200, n_init=200, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=2, random_state=0)
kmeans.fit(X=np.concatenate(np.concatenate((X_train_scaled, X_test_scaled))))
w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
w_in = np.pad(w_in, ((0, 800 - 200), (0, 0)), mode='constant', constant_values=0)

initially_fixed_params = {'hidden_layer_size': 800,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 0.1,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bidirectional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': 1, 'scoring': make_scorer(accuracy_score)}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_km_esn = SeqToLabelESNClassifier(input_to_node=PredefinedWeightsInputToNode(predefined_input_weights=w_in.T),
                                      **initially_fixed_params)

try:
    sequential_search = load("../sequential_search_fsdd_km_sparse_200.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_km_esn, searches=searches).fit(X_train_scaled, y_train)
    dump(sequential_search, "../sequential_search_fsdd_km_sparse_200.joblib")
