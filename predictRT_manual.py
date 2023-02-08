import numpy as np
import pickle
import pandas as pd
import os
import argparse
# Import RandomOverSampler to balance the dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import sys
sys.path.append('../maldi_PIKE/maldi-learn/maldi_learn')
from data import MaldiTofSpectrum
import topf
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='Predicts the ribotype given a Clostridium MALDI-TOF preprocessed data file.')
parser.add_argument('--maldi_path',
                  help='Path to the MALDI data processed by MALDIquant.')
args = parser.parse_args()

# print("Loading MALDI data...")

maldi_path = args.maldi_path
# maldi_path = "./testdata/test_real_24enero"
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
masses = np.vstack(masses)

print("MALDI data loaded.")
print("Total nº of samples: ", str(len(maldis)))


# ============ Load model ===================
print("Loading model...")

models = [ "ksshiba_lin", "ksshiba_rbf", "rf", "favae_vanilla_full", "favae_vanilla_notfull"]
paths = ["results/ksshiba_lin_maldiquant_fulldataset_cpu.pickle", "results/ksshibarbf.pickle","results/rf.pickle", "results/favae_vanilla_maldiquant_fulldataset_cpu.pickle" ]

results = pd.DataFrame({"id": ids})
for p,m in tqdm(zip(paths, models)):
    # Load model
    with open(p, "rb") as handle:
        model = pickle.load(handle)
    # Predict
    print("Predicting with {} model...".format(m))
    if m == "ksshiba_lin" or m == "ksshiba_rbf":

        with open("data/MALDI/data_processedMALDIQuant_noalign.pickle", "rb") as handle:
            original = pickle.load(handle)
        malditrain = original["maldis"] * 1e4
    
        ros = RandomOverSampler()
        malditrain, y = ros.fit_resample(malditrain, original["labels_3cat"])
        if m == 'ksshiba_lin': 
            kernel = "linear"
        else: 
            kernel = "rbf"
        maldis_test = model['model'].struct_data(
                            maldis,
                            method="reg",
                            V=malditrain,
                            kernel=kernel,
                        )
        y_pred, Z_test_mean, Z_test_cov = model['model'].predict([0], [1], maldis_test)
        gray_pred = y_pred['output_view1']['mean_x']
        y_pred_int = np.argmax(gray_pred, axis=1)
        value_pred = np.max(gray_pred, axis=1)

    elif m == "ksshiba_pike":
        with open("data/MALDI/data_processedMALDIQuant_noalign.pickle", "rb") as handle:
            original = pickle.load(handle)
        y3 = original["labels_3cat"]
        malditrain = original["maldis"] * 1e4
        masses_original = np.vstack(original["masses"])
        maldis_pike = [[] for i in range(malditrain.shape[0])]
        transformer= topf.PersistenceTransformer(n_peaks=200)
        for i in tqdm(range(malditrain.shape[0])):
            topf_signal = np.concatenate((masses_original[i, :].reshape(-1, 1), malditrain[i,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            maldis_pike[i] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
        maldis_pike = [MaldiTofSpectrum(maldis_pike[i]) for i in range(len(maldis_pike))]
        maldi_selected = np.array(maldis_pike)
        unique, counts = np.unique(y3, return_counts=True)
        index_of_class0 = np.where(y3==0)[0]
        index_of_class1 = np.where(y3==1)[0]
        index_of_class2 = np.where(y3==2)[0]
        # Randomly select the same number of samples from the class with more samples
        index_of_class0 = np.random.choice(index_of_class0, counts[2], replace=True)
        index_of_class1 = np.random.choice(index_of_class1, counts[2], replace=True)

        # Create a new dataset with the same number of samples for each class
        maldi_resampled = np.concatenate((maldi_selected[index_of_class0], maldi_selected[index_of_class1], maldi_selected[index_of_class2]), axis=0)
        maldis_resampled = [MaldiTofSpectrum(maldi_resampled[i]) for i in range(len(maldi_resampled))]
        
        maldis_piked = [[] for i in range(maldis.shape[0])]
        transformer= topf.PersistenceTransformer(n_peaks=200)
        for i in tqdm(range(maldis.shape[0])):
            topf_signal = np.concatenate((masses[i, :].reshape(-1, 1), maldis[i,:].reshape(-1,1)), axis=1)
            signal_transformed = transformer.fit_transform(topf_signal)
            maldis_piked[i] = MaldiTofSpectrum(signal_transformed[signal_transformed[:,1]>0])
        maldis_test = [MaldiTofSpectrum(maldis_piked[i]) for i in range(len(maldis_piked))]
        
        
        maldis_test = model['model'].struct_data(
                        maldis_test,
                        method="reg",
                        V=maldis_resampled,
                        kernel="pike",
                    )
        y_pred, Z_test_mean, Z_test_cov = model['model'].predict([0], [1], maldis_test)
        gray_pred = y_pred['output_view1']['mean_x']
        y_pred_int = np.argmax(gray_pred, axis=1)
        value_pred = np.max(gray_pred, axis=1)

    elif m == "favae_vanilla_full" or m == "favae_vanilla_notfull" or m =="favae_test":
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

        # Z_mean = model['model'].q_dist.Z['mean']
        # # Project Z_mean and Z_test_mean to 3D using t-SNE
        # tsne = TSNE(n_components=3, random_state=0)
        # Z_mean_3d = tsne.fit_transform(np.vstack((Z_mean, Z_test_mean)))
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # # Train samples
        # ax.scatter(Z_mean_3d[:Z_mean.shape[0], 0], Z_mean_3d[:Z_mean.shape[0], 1], Z_mean_3d[:Z_mean.shape[0], 2], c=np.argmax(model['model'].t[1]['data'], axis=1)[:Z_mean.shape[0]], cmap="Set2", label="Train samples")
        # # Test samples
        # ax.scatter(Z_mean_3d[Z_mean.shape[0]:, 0], Z_mean_3d[Z_mean.shape[0]:, 1], Z_mean_3d[Z_mean.shape[0]:, 2], c=y_pred_int, cmap=['green', 'blue'], marker="s", s=200, label="Test samples")
        # plt.legend()    
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
results["ksshiba_rbf_category"] = results["ksshiba_rbf_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
results["rf_category"] = results["rf_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
results["favae_vanilla_full_category"] = results["favae_vanilla_full_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
results["favae_vanilla_notfull_category"] = results["favae_vanilla_notfull_category"].map({0: "RT027", 1: "RT181", 2: "Others"})
# Cast column id to int
results["id"] = results["id"].astype(int)
# Save results to excel
results.to_excel("results/predictions_ultimatanda73.xlsx", index=False)


# ======================================== Check results in test ========================================
# Load test data from an excel
test = pd.read_excel("testdata/TODAS LAS CEPAS CLOSTRIS.xlsx")

# Select columns from H to K
test = test.iloc[:, 7:9]

# Drop rows with NaN
test.dropna(inplace=True)

# Rename columns: Nuevas para analizar -> id, Unnamed: 8 -> true_category
test.rename(columns={"Nuevas para analizar": "id", "Unnamed: 8": "true_category"}, inplace=True)

# Cast column id to int
test["id"] = test["id"].astype(int)

# Map ribotypes: 027 is RT027, 181 is RT181, all other possible values are Others
test["true_category"] = test["true_category"].map({'027': "RT027", '181': "RT181"})
test["true_category"] = test["true_category"].fillna("Others")


# Merge results with test data
test = test.merge(results, on="id")

# Import accuracy score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Calculate accuracy for each model
for m in models:
    print(m)
    print("Accuracy: ", accuracy_score(test["true_category"], test[m+"_category"]))
    print("Precision: ", precision_score(test["true_category"], test[m+"_category"], average="weighted"))
    print("Recall: ", recall_score(test["true_category"], test[m+"_category"], average="weighted"))
    print("F1: ", f1_score(test["true_category"], test[m+"_category"], average="weighted"))
    print("Confusion matrix: ")
    print(confusion_matrix(test["true_category"], test[m+"_category"]))
    print("")


