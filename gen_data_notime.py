import scipy.io
import numpy as np
import pickle

models = scipy.io.loadmat('./data/models.mat')

Tr_train = np.empty([75, 140])
p_train = np.empty([75, 3])

for iBlock in range(75):
    block = models['models'][0,iBlock]
    Tr_train[iBlock, :] = block[5][:, -1]
    p_train[iBlock, 0] = block[1][0, 0]
    p_train[iBlock, 1] = block[2][0, 0]
    p_train[iBlock, 2] = block[3][0, 0]

u, s, vh = np.linalg.svd(Tr_train.T, full_matrices=True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(Tr_train[:,3])
plt.figure()
plt.semilogy(s)
plt.show()

training_data = {
  "X_train": p_train,
  "Y_train": Tr_train,
}

outfile = open('./data/training_data.p', 'wb')
pickle.dump(training_data, outfile)
outfile.close()
