import scipy.io
import numpy as np
import pickle

ref = scipy.io.loadmat('./data/reference.mat')
print(ref)

Tr_ref = np.empty([201, 140])
p_ref = np.empty([201, 4])

block = ref['Tr']
for iTime in range(201):
    Tr_ref[iTime, :] = block[:, iTime]
    p_ref[iTime, 0] = ref['speed'][0, 0]
    p_ref[iTime, 1] = ref['intensity'][0, 0]
    p_ref[iTime, 2] = ref['direction'][0, 0]
    p_ref[iTime, 3] = iTime/201


import matplotlib.pyplot as plt
plt.figure()
plt.plot(Tr_ref[:,1])
plt.show()

training_data = {
  "X_ref": p_ref,
  "Y_ref": Tr_ref,
}

outfile = open('./data/ref_data.p', 'wb')
pickle.dump(training_data, outfile)
outfile.close()