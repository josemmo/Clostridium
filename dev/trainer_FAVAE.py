from sklearn.model_selection import train_test_split
import numpy as np
import favae as sshiba
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import os
import time
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from data import MaldiTofSpectrum
import topf
# Import tqdm
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# parser = argparse.ArgumentParser(description='SSHIBA IMG VAE with adapative learning over Z latent space')
# parser.add_argument('--lr', type=float, default=1.0,
#                   help='Learning rate value for adapative Z learning. Default = 1 (no adaptative learning)')
# args = parser.parse_args()

# print(args.lr)

lr = 1
# ============ Load data ===================
with open("data/MALDI/data_processedMALDIQuant_noalign.pickle", "rb") as handle:
    data = pickle.load(handle)

maldis_original = data["maldis"] * 1e4
y3_original = data["labels_3cat"]
masses_original = np.array(data["masses"])

# maldis_pike = [[] for i in range(maldis_original.shape[0])]
# transformer= topf.PersistenceTransformer(n_peaks=200)
# for i in tqdm(range(maldis_original.shape[0])):
#     topf_signal = np.concatenate((masses_original[i, :].reshape(-1, 1), maldis_original[i,:].reshape(-1,1)), axis=1)
#     signal_transformed = transformer.fit_transform(topf_signal)
#     maldis_pike[i] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
# maldis_pike = [MaldiTofSpectrum(maldis_pike[i]) for i in range(len(maldis_pike))]

# maldis = np.delete(maldis_original, np.where(y3_original==2), axis=0)
# y3 = np.delete(y3_original, np.where(y3_original==2))
# masses = np.delete(masses_original, np.where(y3_original==2), axis=0)

maldi_selected = maldis_original
y3 = y3_original
masses = masses_original

# ============ Preprocessing ===================

# IF PIKE
# Resample manually to have balanced classes
# Count how many samples of each class
# unique, counts = np.unique(y3, return_counts=True)
# index_of_class0 = np.where(y3==0)[0]
# index_of_class1 = np.where(y3==1)[0]
# index_of_class2 = np.where(y3==2)[0]
# # Randomly select the same number of samples from the class with more samples
# index_of_class0 = np.random.choice(index_of_class0, counts[2], replace=True)
# index_of_class1 = np.random.choice(index_of_class1, counts[2], replace=True)

# # Create a new dataset with the same number of samples for each class
# maldi_resampled = np.concatenate((maldi_selected[index_of_class0], maldi_selected[index_of_class1], maldi_selected[index_of_class2]), axis=0)
# maldis_resampled = [MaldiTofSpectrum(maldi_resampled[i]) for i in range(len(maldi_resampled))]
# y_resampled = np.concatenate((y3[index_of_class0], y3[index_of_class1], y3[index_of_class2]), axis=0)

# OTHERWISE
ros = RandomOverSampler()
maldis_resampled, y_resampled = ros.fit_resample(maldi_selected, y3)

x_train, x_test, y_train, y_test = train_test_split(
    maldis_resampled, y_resampled, train_size=0.6,
)

ohe = OneHotEncoder(sparse=False)
ohe.fit(y3.reshape(-1, 1))
y_train_ohe = ohe.transform(y_train.reshape(-1, 1))
y_test_ohe = ohe.transform(y_test.reshape(-1, 1))

# ============ FAVAE MODEL ===================
print("Creating model")
hyper_parameters = {
    "sshiba": {
        "prune": 1,
        "myKc": 100,
        "pruning_crit": 1e-3,
        "max_it": 10000,
        "latentspace_lr": lr,
    }
}

store = False

myModel_new = sshiba.SSHIBA(
    hyper_parameters["sshiba"]["myKc"],
    hyper_parameters["sshiba"]["prune"],
    latentspace_lr=hyper_parameters["sshiba"]["latentspace_lr"],
)


# MALDI as a VAE
# maldis = myModel_new.struct_data(np.vstack((x_train, x_test)), "vae", latent_dim=20, lr=1e-3, dataset="clostri"
# )
# maldis_test = myModel_new.struct_data(x_test, "vae", latent_dim=20, lr=1e-3, dataset="clostri")

# MALDI as a kernelized version
maldis = myModel_new.struct_data(np.vstack((x_train, x_test)),
    method="reg",
    V=np.vstack((x_train, x_test)),
    kernel="rbf",
)
# maldis_test = myModel_new.struct_data(
#     x_test,
#     method="reg",
#     V=x_train,
#     kernel="linear",
# )

# Labels
labels = myModel_new.struct_data(np.vstack((y_train_ohe, y_test_ohe)), "mult")
# ============ FAVAE TRAINING ===================
print("Training model")
t1 = time.time()
myModel_new.fit(
    maldis,
    labels,
    max_iter=hyper_parameters["sshiba"]["max_it"],
    pruning_crit=hyper_parameters["sshiba"]["pruning_crit"],
    verbose=1,
    store=store,
)
t2 = time.time()

# ============ FAVAE PREDICTION ===================
# Predictions
# y_pred, Z_test_mean, Z_test_cov = myModel_new.predict([0], [1], maldis_test)
# y_pred = y_pred['output_view1']['mean_x']
# y_pred_int = np.argmax(y_pred, axis=1)

# acc = balanced_accuracy_score(y_test, y_pred_int)

# # Import OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# # Instantiate OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# # Transform prediction and test to one hot encoding
# y_pred_ohe = ohe.fit_transform(y_pred_int.reshape(-1, 1))
# y_test_ohe = ohe.transform(y_test.reshape(-1, 1))
# auc = roc_auc_score(y_test_ohe, y_pred_ohe, multi_class='ovr')

# print(acc)
# print(auc)
# print("time: " + str(t2 - t1))

# import seaborn as sns
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # Import TSNE
# from sklearn.manifold import TSNE

# # # Plotting the true labels over the latent space with tSNE
# latent_space = Z_test_mean
# # # Project the latent space to 2D with tSNE
# latent_space_2d = TSNE(n_components=2).fit_transform(latent_space)

# rt027_true = np.where(y_test == 0)[0]
# rt181_true = np.where(y_test == 1)[0]
# rtxxx_true = np.where(y_test == 2)[0]
# rt027_pred = np.where(y_pred_int == 0)[0]
# rt181_pred = np.where(y_pred_int == 1)[0]
# rtxxx_pred = np.where(y_pred_int == 2)[0]

# # Select only missclasified samples
# rt027_miss = np.setdiff1d(rt027_pred, rt027_true)
# rt181_miss = np.setdiff1d(rt181_pred, rt181_true)
# rtxxx_miss = np.setdiff1d(rtxxx_pred, rtxxx_true)

# # Select only correctly classified samples
# rt027_corr = np.intersect1d(rt027_pred, rt027_true)
# rt181_corr = np.intersect1d(rt181_pred, rt181_true)
# rtxxx_corr = np.intersect1d(rtxxx_pred, rtxxx_true)

# plt.figure(figsize=(10, 10))

# # Predictions
# plt.scatter(latent_space_2d[rt027_pred, 0], latent_space_2d[rt027_pred, 1], c='b', label='RT027 prediction')
# plt.scatter(latent_space_2d[rt181_pred, 0], latent_space_2d[rt181_pred, 1], c='orange', label='RT181 prediction')
# plt.scatter(latent_space_2d[rtxxx_pred, 0], latent_space_2d[rtxxx_pred, 1], c='green', label='Others prediction')

# # True values
# plt.scatter(latent_space_2d[rt027_true, 0], latent_space_2d[rt027_true, 1], c='b', marker='x', label='RT027 true')
# plt.scatter(latent_space_2d[rt181_true, 0], latent_space_2d[rt181_true, 1], c='orange', marker='x', label='RT181 true')
# plt.scatter(latent_space_2d[rtxxx_true, 0], latent_space_2d[rtxxx_true, 1], c='green', marker='x', label='Others true')

# plt.ylabel('t-SNE 2')
# plt.xlabel('t-SNE 1')
# plt.legend()
# plt.show()


import pickle
store = {"model": myModel_new}
with open('./results/ksshibarbf.pickle', 'wb') as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
