import scipy.io
import numpy as np
import pickle

models = scipy.io.loadmat('./data/models.mat')

Tr_train = np.empty([201*75, 140])
p_train = np.empty([201*75, 4])

for iBlock in range(75):
    block = models['models'][0,iBlock]
    for iTime in range(201):
        Tr_train[iBlock*201+iTime, :] = block[5][:, iTime]
        p_train[iBlock*201+iTime, 0] = block[1][0, 0]
        p_train[iBlock*201+iTime, 1] = block[2][0, 0]
        p_train[iBlock*201+iTime, 2] = block[3][0, 0]
        p_train[iBlock*201+iTime, 3] = iTime/201

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
