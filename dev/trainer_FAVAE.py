from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import argparse
import favae as sshiba
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score


#parser = argparse.ArgumentParser(description='SSHIBA IMG VAE with adapative learning over Z latent space')
#parser.add_argument('--lr', type=float, default=1.0,
#                   help='Learning rate value for adapative Z learning. Default = 1 (no adaptative learning)')
#args = parser.parse_args()

#print(args.lr)

lr = 1
# ============ Load data ===================
with open('data/MALDI/data_processedMALDIQuant_noalign.pickle', 'rb') as handle:
    data = pickle.load(handle)

maldis = data['maldis']*1e4
y3 = data['labels_3cat']
masses = data['masses']

maldi_selected = maldis
# ============ Preprocessing ===================
ros = RandomOverSampler()
maldis_resampled, y_resampled = ros.fit_resample(maldi_selected, y3)

x_train, x_test, y_train, y_test = train_test_split(maldis_resampled, y_resampled, train_size=0.6)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y3.reshape(-1,1))
y_train_ohe = ohe.transform(y_train.reshape(-1,1))
y_test_ohe = ohe.transform(y_test.reshape(-1,1))

# ============ FAVAE MODEL ===================
print("Creating model")
hyper_parameters = {'sshiba': {"prune": 1, "myKc": 100, "pruning_crit": 1e-3, "max_it": 100000, "latentspace_lr": lr}}

store = False

myModel_new = sshiba.SSHIBA(hyper_parameters['sshiba']['myKc'], hyper_parameters['sshiba']['prune'], latentspace_lr=hyper_parameters['sshiba']['latentspace_lr'])

# Full MALDI data
maldis = myModel_new.struct_data(np.vstack((x_train, x_test)), 'reg')
# Labels
labels = myModel_new.struct_data(y_train_ohe, 'mult')
# ============ FAVAE TRAINING ===================
print("Training model")
myModel_new.fit(maldis, labels,
            max_iter=hyper_parameters['sshiba']['max_it'],
            pruning_crit=hyper_parameters['sshiba']['pruning_crit'],
            verbose=1,
            store=store)

y_pred = np.argmax(myModel_new.t[1]['mean'][x_train.shape[0]:, :], axis=1)
acc = balanced_accuracy_score(y_test, y_pred)

print(acc)


import pickle
store = {"model": myModel_new, "acc": acc, "x_train": x_train, "x_test": x_test, "y_test": y_test, "y_train": y_train}
with open('./results/sshiba_reg_noalignment2.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)

    