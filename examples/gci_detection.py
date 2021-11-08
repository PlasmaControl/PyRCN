import numpy as np
import librosa
import librosa.display
import glob
from pyrcn.echo_state_network import ESNRegressor
from pyrcn.linear_model import IncrementalRegression
from pyrcn.base.blocks import InputToNode, NodeToNode
from sklearn.base import clone
from scipy.signal import find_peaks
from mir_eval import onset
import time
from joblib import Parallel, delayed, dump, load
import csv


from matplotlib import pyplot as plt
#Options
params = {'image.cmap' : 'jet',
          'font.size' : 11,
          'axes.titlesize' : 24,
          'axes.labelsize' : 20,
          'lines.linewidth' : 3,
          'lines.markersize' : 10,
          'xtick.labelsize' : 16,
          'ytick.labelsize' : 16,
          }
plt.rcParams.update(params)
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42


def train_esn(base_input_to_node, base_node_to_node, base_reg, frame_length, file_list):
    print(frame_length)
    try:
        load("models/esn_500u_" + str(frame_length) + ".joblib")
    except FileNotFoundError:
        input_to_node = clone(base_input_to_node)
        node_to_node = clone(base_node_to_node)
        reg = clone(base_reg)
        esn = ESNRegressor(input_to_nodes=[('default', input_to_node)],
                           nodes_to_nodes=[('default', node_to_node)],
                           regressor=reg, random_state=0)
        for file in file_list[:7]:
            X, y = extract_features(file, sr=4000., frame_length=frame_length, target_widening=True)
            esn.partial_fit(X=X, y=y, postpone_inverse=True)
        X , y = extract_features(file_list[8], sr=4000., frame_length=frame_length, target_widening=True)
        esn.partial_fit(X=X, y=y, postpone_inverse=False)
        dump(esn, "esn_500b_" + str(frame_length) + ".joblib")


def extract_features(filename: str, sr: float = 4000., frame_length: int = 81, target_widening: bool = True):
    s, sr = librosa.load(filename, sr=sr, mono=False)
    X = librosa.util.frame(s[0, :], frame_length=frame_length, hop_length=1).T
    y = librosa.util.frame(binarize_signal(s[1, :], 0.04), frame_length=frame_length, hop_length=1).T
    if target_widening:
        return X, np.convolve(y[:, int(frame_length / 2)], [0.5, 1.0, 0.5], 'same')
    else:
        return X, y[:, int(frame_length / 2)]


def binarize_signal(y, thr=0.04):
    y_diff = np.maximum(np.diff(y, prepend=0), thr)
    peaks, _ = find_peaks(y_diff)
    y_bin = np.zeros_like(y_diff, dtype=int)
    y_bin[peaks] = 1
    return y_bin


def peak_picking(y, thr):
    return librosa.util.peak_pick(y, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=thr, wait=10)


def write_output_file(file_name, times, annotations=None):
    with open(file_name, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        if annotations is None:
            for t in times:
                writer.writerow(['%0.3f' % t])
        else:
            for t, lab in zip(times, annotations):
                writer.writerow([('%0.3f' % t), lab])


def write_annotation_file(path, intervals, annotations=None):
    if annotations is not None and len(annotations) != len(intervals):
        raise ParameterError('len(annotations) != len(intervals)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')

        if annotations is None:
            for t_int in intervals:
                writer.writerow(['%0.3f' % t_int[0], '%0.3f' % t_int[1]])
        else:
            for t_int, lab in zip(intervals, annotations):
                writer.writerow(['%0.3f' % t_int[0], '%0.3f' % t_int[1], lab])


def annot_to_time_series(annot, intervals, hop_len):
    times = np.arange(start=0, stop=intervals[-1, 1], step=0.01)
    annotations = np.zeros(shape=times.shape)
    for count, interval in enumerate(intervals):
        annotations[np.multiply(times>=interval[0], times<interval[1])] = annot[count]
    return times, annotations


all_wavs_m = glob.glob(r"C:\Temp\SpLxDataLondonStudents2008\M\*.wav")
print(len(all_wavs_m))
all_wavs_n = glob.glob(r"C:\Temp\SpLxDataLondonStudents2008\N\*.wav")
print(len(all_wavs_n))

base_input_to_node = InputToNode(hidden_layer_size=500, input_activation='identity', k_in=5, input_scaling=14.6,
                                 bias_scaling=0.0, random_state=1)
base_node_to_node = NodeToNode(hidden_layer_size=500, spectral_radius=0.8, leakage=0.5, k_rec=16,
                               bidirectional=True, random_state=1)
base_reg = IncrementalRegression(alpha=1.7e-10)

base_esn = ESNRegressor(input_to_node=base_input_to_node,
                        node_to_node=base_node_to_node,
                        regressor=base_reg)

esn = base_esn
t1 = time.time()
Parallel(n_jobs=1, verbose=50)(delayed(train_esn)(base_input_to_node,base_node_to_node, base_reg,
                                                  frame_length, all_wavs_m)
                               for frame_length in [5, 7, 9, 11, 21, 31, 41, 81])
print("Finished in {0} seconds!".format(time.time() - t1))

exit(0)
