import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from pyrcn.base import InputToNode, PredefinedWeightsInputToNode, NodeToNode
from pyrcn.linear_model import IncrementalRegression
from pyrcn.echo_state_network import SeqToSeqESNClassifier
from pyrcn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from pyrcn.model_selection import SequentialSearchCV

from sklearn.base import clone
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from joblib import dump, load


def read_file(fname,Nfr=-1):
    tmp= open(fname+'.txt', 'rb');a=tmp.read();tmp.close();T=len(a) # Just to know how many frames (T) are there in the file
    if Nfr!=-1:
        T=np.min((T,Nfr))
    dim=[30,30] # Dimension of each frame
    N_fr=dim[0]*dim[1] # size of the input vector
    yuvfile= open(fname+'.yuv', 'rb') # Opening the video file
    door_state_file= open(fname+'.txt', 'rb') # Opening the annotation file
    TARGET=np.zeros((T, 3))
    FRAMES=np.zeros((T,N_fr))
    for t in tqdm(range(T)): # for each frame    
        fr2=np.zeros(N_fr) 
        frame = yuvfile.read(N_fr)
        for i in range(N_fr):
            fr2[i]=frame[i]
        # ----------------------------------    
        fr2=fr2/255.0 # Normalizing the pixel values to [0,1]
        FRAMES[t,:]=fr2
        TARGET[t,int(door_state_file.read(1))] = 1 # setting the desired output class to 1
    return FRAMES, TARGET


try:
    X_train, X_test, y_train, y_test = load(r"E:\RCN_CICSyN2015\Seq_video_dataset_large.joblib")
except FileNotFoundError:
    n_files = 3

    X_total = [None] * n_files
    y_total = [None] * n_files
    n_sequences_total = [None] * n_files
    for k in range(n_files):
        X_total[k], y_total[k] = read_file(r"E:\RCN_CICSyN2015\Seq_" + str(k + 1))

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for k in range(n_files):
        n_sequences_total[k] = int(len(X_total[k]) / 5400)
        X_total[k] = np.array_split(X_total[k], n_sequences_total[k])
        y_total[k] = np.array_split(y_total[k], n_sequences_total[k])
        for m, (X, y) in enumerate(zip(X_total[k], y_total[k])):
            if m < int(.5*n_sequences_total[k]):
                X_train_list.append(X)
                y_train_list.append(y)
            else:
                X_test_list.append(X)
                y_test_list.append(y)

    X_train = np.empty(shape=(len(X_train_list), ), dtype=object)
    y_train = np.empty(shape=(len(y_train_list), ), dtype=object)
    X_test = np.empty(shape=(len(X_test_list), ), dtype=object)
    y_test = np.empty(shape=(len(y_test_list), ), dtype=object)

    for k, (X, y) in enumerate(zip(X_train_list, y_train_list)):
        X_train[k] = X.astype(float)
        y_train[k] = y.astype(int)

    for k, (X, y) in enumerate(zip(X_test_list, y_test_list)):
        X_test[k] = X.astype(float)
        y_test[k] = y.astype(int)
    
    dump([X_train, X_test, y_train, y_test], r"E:\RCN_CICSyN2015\Seq_video_dataset_large.joblib")

print(X_train.shape, X_train[0].shape, y_train.shape, y_train[0].shape)
print(X_test.shape, X_test[0].shape, y_test.shape, y_test[0].shape)

kmeans = MiniBatchKMeans(n_clusters=50, n_init=200, reassignment_ratio=0, max_no_improvement=50, init='k-means++', verbose=2, random_state=0)
kmeans.fit(X=np.concatenate(np.concatenate((X_train, X_test))))
w_in = np.divide(kmeans.cluster_centers_, np.linalg.norm(kmeans.cluster_centers_, axis=1)[:, None])

initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 10,
                          'input_scaling': 0.4,
                          'input_activation': 'identity',
                          'bias_scaling': 0.0,
                          'spectral_radius': 0.0,
                          'leakage': 0.1,
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

kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step3 = {'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}
kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': -1, 'scoring': make_scorer(accuracy_score)}

# The searches are defined similarly to the steps of a sklearn.pipeline.Pipeline:
searches = [('step1', RandomizedSearchCV, step1_esn_params, kwargs_step1),
            ('step2', RandomizedSearchCV, step2_esn_params, kwargs_step2),
            ('step3', GridSearchCV, step3_esn_params, kwargs_step3),
            ('step4', RandomizedSearchCV, step4_esn_params, kwargs_step4)]

base_km_esn = SeqToSeqESNClassifier(input_to_node=PredefinedWeightsInputToNode(predefined_input_weights=w_in.T),
                                    **initially_fixed_params)

try:
    sequential_search = load("../sequential_search_RICSyN2015_km_large.joblib")
except FileNotFoundError:
    sequential_search = SequentialSearchCV(base_km_esn, searches=searches).fit(X_train, y_train)
    dump(sequential_search, "../sequential_search_RICSyN2015_km_large.joblib")

base_esn = clone(sequential_search.best_estimator_)

param_grid = {'hidden_layer_size': [50, 100, 200, 400, 800, 1600],
              'random_state': range(1, 11)}

for params in ParameterGrid(param_grid):
    esn = clone(base_esn).set_params(**params).fit(X=X_train, y=y_train, n_jobs=8)
    y_pred = esn.predict_proba(X_test)
    score = accuracy_score(np.argmax(np.concatenate(y_test), axis=1), np.argmax(np.concatenate(y_pred), axis=1))
    print("ESN with params {0} achieved score of {1}".format(params, score))

