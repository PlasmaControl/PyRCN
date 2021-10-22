import numpy as np
from tqdm import tqdm

from pyrcn.base import InputToNode, PredefinedWeightsInputToNode, NodeToNode
from pyrcn.echo_state_network import SeqToSeqESNClassifier
from pyrcn.metrics import mean_squared_error

from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
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


initially_fixed_params = {'hidden_layer_size': 50,
                          'k_in': 10,
                          'input_activation': 'identity',
                          'k_rec': 10,
                          'reservoir_activation': 'tanh',
                          'bi_directional': False,
                          'wash_out': 0,
                          'continuation': False,
                          'alpha': 1e-3,
                          'random_state': 42}

step0_esn_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                    'spectral_radius': uniform(loc=0, scale=2),
                    'leakage': loguniform(1e-5, 1e0),
                    'bias_scaling': uniform(loc=0, scale=2)}

kwargs_step0 = {'n_iter': 1000, 'random_state': 42, 'verbose': 1, 'n_jobs': 1, 'scoring': make_scorer(mean_squared_error, greater_is_better=False, needs_proba=True)}

base_esn = SeqToSeqESNClassifier(**initially_fixed_params)

try:
    random_search_basic = load("../random_search_RICSyN2015_basic.joblib")
except FileNotFoundError:
    random_search_basic = RandomizedSearchCV(estimator=base_esn, param_distributions=step0_esn_params,
                                             **kwargs_step0).fit(X_train, y_train)
    dump(random_search_basic, "../random_search_RICSyN2015_basic.joblib")
