import numpy as np
import pickle
import pandas as pd
import os
from sklearn.metrics import balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser(description='Predicts the ribotype given a Clostridium MALDI-TOF preprocessed data file.')
parser.add_argument('--maldi_path', type=float, default=1.0,
                  help='Path to the MALDI data processed by MALDIquant.')
args = parser.parse_args()

print("Loading MALDI data...")

maldi_path = args.maldi_path
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

print("MALDI data loaded.")
print("Total nº of samples: ", str(len(maldis)))
