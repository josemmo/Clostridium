import numpy as np
import pickle
import pandas as pd
import os
import argparse
# Import RandomOverSampler to balance the dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("=============================================================")
print("Creating folders to store information...")
try:
    os.makedirs("./results/images")
except FileExistsError:
    pass

print("Images will be stored at ./results/images")
print("Predictions will be stored as an Excel file at ./results/")
print("MALDI-TOFs processed by MALDIquant will be stored at ./results/data_maldiquant")
print("=============================================================")
print("Loading MALDI data...")

maldi_path = "./results/data_maldiquant"
# READ DATA MZML
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(maldi_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

masses = []
maldis = []
ids = []
# CONVERT TO A PANDAS DATAFRAME
for filepath in listOfFiles:
    file = filepath.split("\\")[-1]
    if file == ".DS_Store":
        continue
    m = pd.read_csv(filepath)
    maldis.append(m["intensity"].values[0:18000])
    masses.append(m["mass"].values[0:18000])
    ids.append(filepath.split("/")[-2].split(" ")[-1])
maldis = np.vstack(maldis)*1e4

print("MALDI data loaded.")
print("Total nº of samples: ", str(len(maldis)))


# ============ Load model ===================
print("Loading model...")

models = [ "ksshiba_lin", "rf", "favae_vanilla"]
paths = [ "results/ksshiba_lin_maldiquant_fulldataset_cpu.pickle", "results/rf.pickle", "results/favae_vanilla_maldiquant_fulldataset_cpu.pickle", ]
# models = ["ksshiba_lin", "rf"]
# paths = ["results/ksshibalin.pickle", "results/rf.pickle"]

results = pd.DataFrame({"id": ids})
for p,m in zip(paths, models):
    # Load model
    with open(p, "rb") as handle:
        model = pickle.load(handle)
    # Predict
    print("Predicting with {} model...".format(m))
    if m == "ksshiba_lin":

        with open("data/MALDI/data_processedMALDIQuant_noalign.pickle", "rb") as handle:
            original = pickle.load(handle)
        malditrain = original["maldis"] * 1e4
    
        ros = RandomOverSampler()
        maldis_resampled, y_resampled = ros.fit_resample(malditrain, original["labels_3cat"])
        maldis_test = model['model'].struct_data(
                        maldis,
                        method="reg",
                        V=maldis_resampled,
                        kernel="linear",
                    )
        y_pred, Z_test_mean, Z_test_cov = model['model'].predict([0], [1], maldis_test)
        gray_pred = y_pred['output_view1']['mean_x']
        y_pred_int = np.argmax(gray_pred, axis=1)
        value_pred = np.max(gray_pred, axis=1)
    elif m == "favae_vanilla" or m == "favae_1d":
        maldis_test = model['model'].struct_data(
                        maldis,
                        method="vae",
                        latent_dim=20,
                        lr=1e-3,
                        dataset="clostri",
                    )
        y_pred, Z_test_mean, Z_test_cov = model['model'].predict([0], [1], maldis_test)
        gray_pred = y_pred['output_view1']['mean_x']
        y_pred_int = np.argmax(gray_pred, axis=1)
        value_pred = np.max(gray_pred, axis=1)

        Z_mean = model['model'].q_dist.Z['mean']
        # Project Z_mean and Z_test_mean to 3D using t-SNE
        tsne = TSNE(n_components=3, random_state=0)
        Z_mean_3d = tsne.fit_transform(np.vstack((Z_mean, Z_test_mean)))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Train samples
        ax.scatter(Z_mean_3d[:Z_mean.shape[0], 0], Z_mean_3d[:Z_mean.shape[0], 1], Z_mean_3d[:Z_mean.shape[0], 2], c=np.argmax(model['model'].t[1]['data'], axis=1)[:Z_mean.shape[0]], cmap="Set2", label="Train samples")
        # Test samples
        ax.scatter(Z_mean_3d[Z_mean.shape[0]:, 0], Z_mean_3d[Z_mean.shape[0]:, 1], Z_mean_3d[Z_mean.shape[0]:, 2], c=y_pred_int, cmap=['green', 'blue'], marker="s", s=200, label="Test samples")
        plt.legend()    
        # Plot

        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=3, random_state=0)
        # Z_mean_2d = tsne.fit_transform(np.vstack((Z_mean, Z_test_mean)))
        # # Plot
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # plt.figure(figsize=(10, 10))
        # # Train samples

        # sns.scatterplot(x=Z_mean_2d[:Z_mean.shape[0], 0], y=Z_mean_2d[:Z_mean.shape[0], 1], hue=np.argmax(model['model'].t[1]['data'], axis=1)[:Z_mean.shape[0]], palette="Set2", label="Train samples")
        # # Test samples
        # sns.scatterplot(x=Z_mean_2d[Z_mean.shape[0]:, 0], y=Z_mean_2d[Z_mean.shape[0]:, 1], hue=y_pred_int, palette=['green', 'blue'], marker="s", s=200, label="Test samples")


    elif m == "rf":
        y_pred_int = model['model'].predict(maldis)
        value_pred = np.max(model['model'].predict_proba(maldis), axis=1)

    print("Saving results...")
    results[m+'_category'] = y_pred_int
    results[m+'_probability'] = value_pred


# Map ribotypes: 0 is RT027, 1 is RT181, 2 is Others
results["ksshiba_lin_category"] = results["ksshiba_lin_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
results["rf_category"] = results["rf_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
results["favae_vanilla_category"] = results["favae_vanilla_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
# Save results to excel
results.to_excel("results/predictions_ultimatanda73.xlsx", index=False)


