import numpy as np
from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

# load the mnist dataset
X, y = fetch_openml(data_id=554, return_X_y=True, cache=True)

# find first digit occurrences
idx = np.ones((10,))*-1
cnt = 0
while np.any(idx == -1):
    if idx[int(y[cnt])] == -1.0:
        idx[int(y[cnt])] = int(cnt)
    cnt += 1

# display digits
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.resize(X[int(idx[i])], (28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
