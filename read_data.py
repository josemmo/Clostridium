import pandas as pd
import numpy as np
import os

results = {}

maldi_path = "data/MALDI/allavgtogether_noalign"
# READ DATA MZML
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(maldi_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

masses = []
maldis = []
ids = []
# CONVERT TO A PANDAS DATAFRAME
for filepath in listOfFiles:
    file = filepath.split("/")[-1]
    if file == ".DS_Store":
        continue
    m = pd.read_csv(filepath)
    maldis.append(m["intensity"].values[0:18000])
    masses.append(m["mass"].values[0:18000])
    ids.append(filepath.split("/")[-2].split(" ")[-1])
maldis = np.vstack(maldis)

labels=[]
for i in ids:
    labels.append(int(i.split("-")[np.where(np.asarray([len(x) for x in i.split("-")])==3)[0][0]]))
labels = np.asarray(labels)

y3 = np.arange(275)
y3[np.where(labels==27)[0]]=0
y3[np.where(labels==181)[0]]=1
y3[np.where((labels!=181) & (labels!=27))[0]]=2

import pickle

data = {'maldis': maldis, 'masses': masses, 'ids': ids,'labels_raw':labels, 'labels_3cat': y3}
with open('./data/MALDI/data_processedMALDIQuant_noalign.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)