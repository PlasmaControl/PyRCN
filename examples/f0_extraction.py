import os
import glob
import numpy as np
from tqdm import tqdm
import time
import librosa

from joblib import dump, load

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV, RandomizedSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import make_scorer
from pyrcn.metrics import mean_squared_error
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.util import FeatureExtractor
from pyrcn.datasets import fetch_ptdb_tug_dataset
from pyrcn.echo_state_network import SeqToSeqESNRegressor
from pyrcn.base import InputToNode, PredefinedWeightsInputToNode, NodeToNode
import matplotlib.pyplot as plt


def create_feature_extraction_pipeline(sr=16000, frame_sizes=[512, 1024]):
    audio_loading = Pipeline([("load_audio", FeatureExtractor(librosa.load, sr=sr, mono=True)),
                              ("normalize", FeatureExtractor(librosa.util.normalize, norm=np.inf))])
    

    spectrograms = FeatureExtractor(librosa.feature.melspectrogram, sr=sr, n_fft=frame_sizes[0], 
                                    hop_length=160, window='hann', center=False, power=2.0, n_mels=80, 
                                    fmin=40, fmax=4000, htk=True)
    feature_extractor = Pipeline([("spectrograms", spectrograms),
                                  ("power_to_db", FeatureExtractor(librosa.power_to_db, ref=1))])

    feature_extraction_pipeline = Pipeline([("audio_loading", audio_loading),
                                            ("feature_extractor", feature_extractor)])
    return feature_extraction_pipeline


# Load and preprocess the dataset
feature_extraction_pipeline = create_feature_extraction_pipeline()

X_train, X_test, y_train, y_test = fetch_ptdb_tug_dataset(data_origin="Z:/Projekt-Pitch-Datenbank/SPEECH_DATA", 
                                                          data_home=None, preprocessor=feature_extraction_pipeline, 
                                                          force_preprocessing=False)

X_train, y_train = shuffle(X_train, y_train, random_state=0)

def tsplot(ax, data,**kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
    ax.margins(x=0)


fig, axs = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(4, 2.5)
tsplot(axs[0], np.concatenate(np.hstack((X_train, X_test))))
axs[0].set_xlabel('Feature Index')
axs[0].set_ylabel('Magnitude')

# All features carry information, since the variance is always large
# We can fit a StandardScaler here!
scaler = StandardScaler().fit(np.concatenate(X_train))
for k, X in enumerate(X_train):
    X_train[k] = scaler.transform(X=X)
for k, X in enumerate(X_test):
    X_test[k] = scaler.transform(X=X)

tsplot(axs[1], np.concatenate(np.hstack((X_train, X_test))))
axs[1].set_xlabel('Feature Index')
axs[1].set_ylabel('Magnitude')
plt.grid()
plt.tight_layout()
plt.show()
# Define several error functions for $f_{0}$ extraction

# In[ ]:


def gpe(y_true, y_pred):
    """
    Gross pitch error:
    
    All frames that are considered voiced by both pitch tracker and ground truth, 
    for which the relative pitch error is higher than a certain threshold (\SI{20}{\percent}).
    
    """
    idx = np.nonzero(y_true*y_pred)[0]
    return np.mean(np.abs(y_true[idx] - y_pred[idx]) > 0.2 * y_true[idx])


def vde(y_true, y_pred):
    """
    Voicing Decision Error:
    
    Proportion of frames for which an incorrect voiced/unvoiced decision is made.
    
    """
    return zero_one_loss(y_true, y_pred)


def fpe(y_true, y_pred):
    """
    Fine Pitch Error:
    
    Standard deviation of the distribution of relative error values (in cents) from the frames
    that do not have gross pitch errors
    """
    idx_voiced = np.nonzero(y_true * y_pred)[0]
    idx_correct = np.argwhere(np.abs(y_true - y_pred) <= 0.2 * y_true).ravel()
    idx = np.intersect1d(idx_voiced, idx_correct)
    if idx.size == 0:
        return 0
    else:
        return 100 * np.std(np.log2(y_pred[idx] / y_true[idx]))


def ffe(y_true, y_pred):
    """
    $f_{0}$ Frame Error:
    
    Proportion of frames for which an error (either according to the GPE or the VDE criterion) is made.
    FFE can be seen as a single measure for assessing the overall performance of a pitch tracker.
    """
    idx_correct = np.argwhere(np.abs(y_true - y_pred) <= 0.2 * y_true).ravel()
    return 1 - len(idx_correct) / len(y_true)


def custom_scorer(y_true, y_pred):
    gross_pitch_error = [None] * len(y_true)
    for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        gross_pitch_error[k] = gpe(y_true=y_t[:, 0]*y_t[:, 1], y_pred=y_p[:, 0]*(y_p[:, 1] >= .5))
    return np.mean(gross_pitch_error)

gpe_scorer = make_scorer(custom_scorer, greater_is_better=False)



# Set up a ESN
# To develop an ESN model for f0 estimation, we need to tune several hyper-parameters, e.g., input_scaling, spectral_radius, bias_scaling and leaky integration.
# We follow the way proposed in the paper for multipitch tracking and for acoustic modeling of piano music to optimize hyper-parameters sequentially.
# We define the search spaces for each step together with the type of search (a grid search in this context).
# At last, we initialize a SeqToSeqESNRegressor with the desired output strategy and with the initially fixed parameters.

initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage':1.0,
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bi_directional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step1_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2)}

step2_esn_params = {'leakage': loguniform(1e-5, 1e0)}
step3_esn_params = {'bias_scaling': np.linspace(0.0, 1.0, 11)}
step4_esn_params = {'alpha': loguniform(1e-5, 1e1)}

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': 1, 'scoring': gpe_scorer}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': gpe_scorer}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': gpe_scorer}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': gpe_scorer}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_esn = SeqToSeqESNRegressor(**initially_fixed_params)


# We provide a SequentialSearchCV that basically iterates through the list of searches that we have defined before. It can be combined with any model selection tool from scikit-learn.
try: 
    sequential_search = load("../sequential_search_f0_mel.joblib")
except FileNotFoundError:
    print(FileNotFoundError)
    sequential_search = SequentialSearchCV(base_esn, searches=searches).fit(X_train[:10], y_train[:10])
    dump(sequential_search, "sequential_search_f0_mel.joblib")


# Visualize hyperparameter optimization

df = pd.DataFrame(sequential_search.all_cv_results_["step1"])
pvt = pd.pivot_table(df, values='mean_test_score', index='param_input_scaling', columns='param_spectral_radius')

pvt.columns = pvt.columns.astype(float)
pvt2 =  pd.DataFrame(pvt.loc[pd.IndexSlice[0:1], pd.IndexSlice[0.0:1.0]])

fig = plt.figure()
ax = sns.heatmap(pvt2, xticklabels=pvt2.columns.values.round(2), yticklabels=pvt2.index.values.round(2), cbar_kws={'label': 'Score'})
ax.invert_yaxis()
plt.xlabel("Spectral Radius")
plt.ylabel("Input Scaling")
fig.set_size_inches(4, 2.5)
tick_locator = ticker.MaxNLocator(10)
ax.yaxis.set_major_locator(tick_locator)
ax.xaxis.set_major_locator(tick_locator)
# plt.savefig('optimize_is_sr.pdf', bbox_inches='tight', pad_inches=0)

df = pd.DataFrame(sequential_search.all_cv_results_["step2"])
fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.lineplot(data=df, x="param_leakage", y="mean_test_score")
plt.xlabel("Leakage")
plt.ylabel("Score")
# plt.xlim((0, 1))
tick_locator = ticker.MaxNLocator(5)
ax.xaxis.set_major_locator(tick_locator)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
plt.grid()
# plt.savefig('optimize_leakage.pdf', bbox_inches='tight', pad_inches=0)

df = pd.DataFrame(sequential_search.all_cv_results_["step3"])
fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.lineplot(data=df, x="param_bias_scaling", y="mean_test_score")
plt.xlabel("Bias Scaling")
plt.ylabel("Score")
plt.xlim((0, 2))
tick_locator = ticker.MaxNLocator(5)
ax.xaxis.set_major_locator(tick_locator)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
plt.grid()
# plt.savefig('optimize_bias_scaling.pdf', bbox_inches='tight', pad_inches=0)

df = pd.DataFrame(sequential_search.all_cv_results_["step4"])
fig = plt.figure()
fig.set_size_inches(2, 1.25)
ax = sns.lineplot(data=df, x="param_beta", y="mean_test_score")
plt.xlabel("Beta")
plt.ylabel("Score")
plt.xlim((0, 2))
tick_locator = ticker.MaxNLocator(5)
ax.xaxis.set_major_locator(tick_locator)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
plt.grid()
# plt.savefig('optimize_beta.pdf', bbox_inches='tight', pad_inches=0)


gpe_training = [None] * len(training_files)
vde_training = [None] * len(training_files)
fpe_training = [None] * len(training_files)
ffe_training = [None] * len(training_files)
with tqdm(total=len(training_files)) as pbar:
    for k, file_name in enumerate(training_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_training[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_training[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_training[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_training[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

gpe_validation = [None] * len(validation_files)
vde_validation = [None] * len(validation_files)
fpe_validation = [None] * len(validation_files)
ffe_validation = [None] * len(validation_files)
with tqdm(total=len(validation_files)) as pbar:
    for k, file_name in enumerate(validation_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_validation[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_validation[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_validation[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_validation[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

gpe_test = [None] * len(test_files)
vde_test = [None] * len(test_files)
fpe_test = [None] * len(test_files)
ffe_test = [None] * len(test_files)
with tqdm(total=len(test_files)) as pbar:
    for k, file_name in enumerate(test_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_test[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_test[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_test[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_test[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

print("Training: GPE\t VDE\t FPE\t FFE")
print("Training: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_training), np.mean(vde_training), np.mean(fpe_training), np.mean(ffe_training) ))
print("Validation: GPE\t VDE\t FPE\t FFE")
print("Validation: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_validation), np.mean(vde_validation), np.mean(fpe_validation), np.mean(ffe_validation) ))
print("Test: GPE\t VDE\t FPE\t FFE")
print("Test: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_test), np.mean(vde_test), np.mean(fpe_test), np.mean(ffe_test) ))


# Find the negative examples from training, validation and test sets

# In[ ]:


np.argmax(gpe_training), np.argmax(gpe_validation), np.argmax(gpe_test)


# Find the positive examples from training, validation and test sets

# In[ ]:


np.argmin(gpe_training), np.argmin(gpe_validation), np.argmin(gpe_test)


# Visualize worst and best training example

# In[ ]:


X, y = extract_features(training_files[323])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(training_files[303])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# Visualize worst and best validation example

# In[ ]:


X, y = extract_features(validation_files[33])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(validation_files[243])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# Visualize worst and best test example

# In[ ]:


X, y = extract_features(test_files[21])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(test_files[368])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# $K$-Means initialization

# In[ ]:


t1 = time.time()
kmeans = MiniBatchKMeans(n_clusters=500, n_init=20, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=1, random_state=0)
print("Fitting kmeans with features from the training set...")
X = [None] * len(training_files)
y = [None] * len(training_files)
with tqdm(total=len(training_files)) as pbar:
    for k, file_name in enumerate(training_files):
        X[k], y[k] = extract_features(file_name)
        pbar.update(1)
    kmeans.fit(X=np.vstack(X))
print("done in {0}!".format(time.time() - t1))
del X
del y


# Initialize an Echo State Network

# In[ ]:


if base_input_to_nodes.hidden_layer_size <=500:
    w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])
else:
    w_in = np.pad(np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None]), ((0, base_input_to_nodes.hidden_layer_size - 500), (0, 0)), mode='constant', constant_values=0)

base_input_to_node = PredefinedWeightsInputToNode(predefined_input_weights=w_in.T, activation='identity', input_scaling=0.1)
base_node_to_node = NodeToNode(hidden_layer_size=500, spectral_radius=0.1, leakage=1.0, bias_scaling=2.1, k_rec=10, random_state=10)
base_reg = FastIncrementalRegression(alpha=1e-3)

base_esn = ESNRegressor(input_to_node=base_input_to_node,
                        node_to_node=base_node_to_node,
                        regressor=base_reg)


# Try to load a pre-trained ESN

# In[ ]:


try:
    esn = load("dataset/f0_extraction/models/kmeans_esn_500.joblib")
except FileNotFoundError:
    print("Fitting ESN with features from the training set...")
    esn = base_esn
    with tqdm(total=len(training_files)) as pbar:
        for k, file_name in enumerate(training_files[:-1]):
            X, y = extract_features(file_name)
            esn.partial_fit(X=X, y=y, postpone_inverse=True)
            pbar.update(1)
        X, y = extract_features(training_files[-1])
        esn.partial_fit(X=X, y=y, postpone_inverse=False)
        pbar.update(1)
    print("done!")
    dump(esn, "kmeans_esn_500.joblib")


# Compute errors on the training, validation and test set

# In[ ]:


gpe_training = [None] * len(training_files)
vde_training = [None] * len(training_files)
fpe_training = [None] * len(training_files)
ffe_training = [None] * len(training_files)
with tqdm(total=len(training_files)) as pbar:
    for k, file_name in enumerate(training_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_training[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_training[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_training[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_training[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

gpe_validation = [None] * len(validation_files)
vde_validation = [None] * len(validation_files)
fpe_validation = [None] * len(validation_files)
ffe_validation = [None] * len(validation_files)
with tqdm(total=len(validation_files)) as pbar:
    for k, file_name in enumerate(validation_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_validation[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_validation[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_validation[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_validation[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

gpe_test = [None] * len(test_files)
vde_test = [None] * len(test_files)
fpe_test = [None] * len(test_files)
ffe_test = [None] * len(test_files)
with tqdm(total=len(test_files)) as pbar:
    for k, file_name in enumerate(test_files):
        X, y = extract_features(file_name)
        y_pred = esn.predict(X=X)
        gpe_test[k] = gpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        vde_test[k] = vde(y_true=y[:, 1], y_pred=y_pred[:, 1] >= .5)
        fpe_test[k] = fpe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        ffe_test[k] = ffe(y_true=y[:, 0]*y[:, 1], y_pred=y_pred[:, 0]*(y_pred[:, 1] >= .5))
        pbar.update(1)

print("Training: GPE\t VDE\t FPE\t FFE")
print("Training: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_training), np.mean(vde_training), np.mean(fpe_training), np.mean(ffe_training) ))
print("Validation: GPE\t VDE\t FPE\t FFE")
print("Validation: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_validation), np.mean(vde_validation), np.mean(fpe_validation), np.mean(ffe_validation) ))
print("Test: GPE\t VDE\t FPE\t FFE")
print("Test: {0}\t {1}\t {2}\t {3}".format(np.mean(gpe_test), np.mean(vde_test), np.mean(fpe_test), np.mean(ffe_test) ))


# Find the negative examples from training, validation and test sets

# In[ ]:


np.argmax(gpe_training), np.argmax(gpe_validation), np.argmax(gpe_test)


# Find the positive examples from training, validation and test sets

# In[ ]:


np.argmin(gpe_training), np.argmin(gpe_validation), np.argmin(gpe_test)


# Visualize worst and best training example

# In[ ]:


X, y = extract_features(training_files[1547])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(training_files[7])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# Visualize worst and best validation example

# In[ ]:


X, y = extract_features(validation_files[277])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(validation_files[499])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# Visualize worst and best test example

# In[ ]:


X, y = extract_features(test_files[21])
y_pred = esn.predict(X=X)
plt.subplot(2,1,1)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))
X, y = extract_features(test_files[276])
y_pred = esn.predict(X=X)
plt.subplot(2,1,2)
plt.plot(y[:, 0])
plt.plot(y_pred[:, 0]*(y_pred[:, 1] >= .5))


# In[ ]:




