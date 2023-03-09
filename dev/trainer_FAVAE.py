from sklearn.model_selection import train_test_split
import numpy as np
import favae as sshiba
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
import os
import time
import sys
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import pickle
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="config.yml",
    help="Path to config file",
)
parser.add_argument("--kernel_pike", type=bool, default=False, help="Use PIKE kernel")
parser.add_argument("--vae", type=bool, default=False, help="Use VAE")
parser.add_argument(
    "--semisupervised", type=bool, default=False, help="Use semi-supervised"
)
parser.add_argument("--prune", type=bool, default=True, help="Use pruning")
parser.add_argument("--Kc", type=int, default=10, help="Number of latent dimensions")
parser.add_argument("--prune_th", type=float, default=0.001, help="Pruning threshold")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

args = parser.parse_args()


# ============ Parameters ===================
print("Loading config")
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

main_path = config["main_path"]
path_to_PIKE = main_path + "maldi_PIKE/maldi-learn/maldi_learn"
maldi_data_path = main_path + "data/old/MALDI/data_processedMALDIQuant_noalign.pickle"

kernel_pike = args.kernel_pike
vae_maldi = args.vae
semisupervised = args.semisupervised
sshiba_hyperparams = {
    "prune": args.prune,
    "myKc": args.Kc,
    "pruning_crit": args.prune_th,
    "lr": args.lr,
    "max_it": args.epochs,
}
store = config["store"]
verbose = config["verbose"]
# Path to store results is results/ and the name of the pkl is each sshiba hyperparameter
path_to_store = (
    main_path
    + "results/pike_"
    + str(kernel_pike)
    + "_vae_"
    + str(vae_maldi)
    + "_semisupervised_"
    + str(semisupervised)
    + ".pkl"
)

# ============ Load data ===================
print("Loading data...")
with open(maldi_data_path, "rb") as handle:
    data = pickle.load(handle)

maldis_original = data["maldis"] * 1e4
y3 = data["labels_3cat"]
masses_original = np.array(data["masses"])

# ============ PIKE ===================
if kernel_pike:
    print("Constructing kernel... ")
    sys.path.append("../maldi_PIKE/maldi-learn/maldi_learn")
    from data import MaldiTofSpectrum
    import topf

    maldis_pike = [[] for i in range(maldis_original.shape[0])]
    transformer = topf.PersistenceTransformer(n_peaks=200)
    for i in tqdm(range(maldis_original.shape[0])):
        topf_signal = np.concatenate(
            (
                masses_original[i, :].reshape(-1, 1),
                maldis_original[i, :].reshape(-1, 1),
            ),
            axis=1,
        )
        signal_transformed = transformer.fit_transform(topf_signal)
        maldis_pike[i] = MaldiTofSpectrum(
            signal_transformed[signal_transformed[:, 1] > 0]
        )
    maldis_pike = [MaldiTofSpectrum(maldis_pike[i]) for i in range(len(maldis_pike))]
    maldis_original = maldis_pike


# ============ Preprocessing ===================

# Split train and test
print("Splitting train and test...")
x_train, x_test, y_train, y_test = train_test_split(
    maldis_original,
    y3,
    train_size=0.6,
)

print("Preprocessing data...")
print("Resampling...")
if kernel_pike:
    # Resample manually to have balanced classes
    # Count how many samples of each class
    unique, counts = np.unique(y3, return_counts=True)
    index_of_class0 = np.where(y3 == 0)[0]
    index_of_class1 = np.where(y3 == 1)[0]
    index_of_class2 = np.where(y3 == 2)[0]
    # Randomly select the same number of samples from the class with more samples
    index_of_class0 = np.random.choice(index_of_class0, counts[2], replace=True)
    index_of_class1 = np.random.choice(index_of_class1, counts[2], replace=True)

    # Create a new dataset with the same number of samples for each class
    maldi_resampled = np.concatenate(
        (
            maldis_original[index_of_class0],
            maldis_original[index_of_class1],
            maldis_original[index_of_class2],
        ),
        axis=0,
    )
    maldis_resampled = [
        MaldiTofSpectrum(maldi_resampled[i]) for i in range(len(maldi_resampled))
    ]
    y_resampled = np.concatenate(
        (y3[index_of_class0], y3[index_of_class1], y3[index_of_class2]), axis=0
    )
else:
    ros = RandomOverSampler()
    x_train, y_train = ros.fit_resample(x_train, y_train)


print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# Convert to one hot encoding the labels
print("Converting to one hot encoding...")
ohe = OneHotEncoder(sparse=False)
ohe.fit(y3.reshape(-1, 1))
y_train_ohe = ohe.transform(y_train.reshape(-1, 1))
y_test_ohe = ohe.transform(y_test.reshape(-1, 1))

# ============ FAVAE MODEL ===================
print("Creating model..")
# Define the model
myModel_new = sshiba.SSHIBA(
    sshiba_hyperparams["myKc"],
    sshiba_hyperparams["prune"],
    latentspace_lr=sshiba_hyperparams["lr"],
)

if vae_maldi:
    # MALDI as a VAE
    if semisupervised:
        print("VAE semisupervised")
        maldis = myModel_new.struct_data(
            np.vstack((x_train, x_test)),
            "vae",
            latent_dim=20,
            lr=1e-3,
            dataset="clostri",
        )
    else:
        print("VAE supervised")
        maldis = myModel_new.struct_data(
            x_train, "vae", latent_dim=20, lr=1e-3, dataset="clostri"
        )
        maldis_test = myModel_new.struct_data(
            x_test, "vae", latent_dim=20, lr=1e-3, dataset="clostri"
        )
elif kernel_pike:
    # MALDI as a kernelized version
    if semisupervised:
        print("PIKE semisupervised")
        maldis = myModel_new.struct_data(
            np.vstack((x_train, x_test)),
            method="reg",
            V=np.vstack((x_train, x_test)),
            kernel="pike",
        )
    else:
        print("PIKE supervised")
        maldis = myModel_new.struct_data(
            x_train,
            method="reg",
            V=x_tain,
            kernel="pike",
        )
        maldis_test = myModel_new.struct_data(
            x_test, method="reg", V=x_train, kernel="pike"
        )
else:
    # Raise not implemented error
    raise NotImplementedError("Only VAE and PIKE kernels are implemented")

# Labels
labels = myModel_new.struct_data(y_train_ohe, "mult")
# ============ FAVAE TRAINING ===================
print("Training model...")

print("Maldis shape: ", maldis["data"].shape)
print("Labels shape: ", labels["data"].shape)
myModel_new.fit(
    maldis,
    labels,
    max_iter=sshiba_hyperparams["max_it"],
    pruning_crit=sshiba_hyperparams["pruning_crit"],
    verbose=verbose,
    store=store,
)


# ============ FAVAE PREDICTION ===================
print("Predicting...")
if semisupervised:
    y_pred = myModel_new.t[1][x_train.shape[0] :, :]
    y_pred_int = np.argmax(y_pred, axis=1)
else:
    # Predictions
    y_pred, Z_test_mean, Z_test_cov = myModel_new.predict([0], [1], maldis_test)
    y_pred = y_pred["output_view1"]["mean_x"]
    y_pred_int = np.argmax(y_pred, axis=1)

# ============ FAVAE EVALUATION ===================
bacc = balanced_accuracy_score(y_test, y_pred_int)

auc = roc_auc_score(y_test_ohe, y_pred, multi_class="ovr")

cm = confusion_matrix(y_test, y_pred_int)

print("Balanced accuracy: ", bacc)
print("AUC: ", auc)

# Store them in a dictionary where the main key is the method used (pike or VAE):
if kernel_pike:
    method = "pike"
else:
    method = "vae"


results = {
    "method": method,
    "bacc": bacc,
    "auc": auc,
}


storing = {"results": results, "model": myModel_new}
with open(path_to_store, "wb") as handle:
    pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
