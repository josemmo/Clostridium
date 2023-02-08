import tempVAE
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

with open('data/MALDI/data_processedMALDIQuant_noalign.pickle', 'rb') as handle:
    data = pickle.load(handle)

maldis_original = data['maldis']*1e4
y3_original = data['labels_3cat']
masses_original = data['masses']

maldis = np.delete(maldis_original, np.where(y3_original==2), axis=0)
y3 = np.delete(y3_original, np.where(y3_original==2))
masses = np.delete(masses_original, np.where(y3_original==2), axis=0)

p027 = np.where(y3==0)[0]
p181 = np.where(y3==0)[1]

maldis_tensor = torch.Tensor(np.expand_dims(maldis, axis=1))
vae = tempVAE.SignalVAE(dataset='clostri')

vae.trainloop(maldis_tensor, epochs=100)

vae.latentdim=2
mean, cov = vae.update_x(maldis_tensor)
rec_sign = vae.reconstruction(mean, cov)

plt.figure()
plt.scatter(mean[p027, 0], mean[p027, 1], color='red', label="RT 027")
plt.scatter(mean[p181, 0], mean[p181, 1], color='black', label="RT 181")
plt.show()