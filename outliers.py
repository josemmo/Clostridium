import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import yaml
import pickle
import scipy.stats as stats

config = "config.yaml"

# Load the data
print("Loading config")
with open(
    "/export/usuarios01/alexjorguer/Datos/HospitalProject/Clostridium/" + config
) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

main_path = config["main_path"]
maldi_path = main_path + "data/data_final.pkl"
results = main_path + "results_paper/"

# ============ Load data ===================
print("Loading data...")
with open(maldi_path, "rb") as handle:
    data = pickle.load(handle)

x = data["intensities"] * 1e4
ids = data["ids"].squeeze()
masses = data["masses"]
y = data["labels"]

mean_signals = np.mean(x, axis=1)
# Detect outliers in X using the z-score
z = np.abs(stats.zscore(mean_signals))
threshold = 2.5
outliers = np.where(z > threshold)
print("Outliers detected: ", len(outliers[0]))
# Print ids of outliers
print("Ids of outliers: ", ids[outliers])

# Plot possible outliers given by the doctors
possible_outliers = [18173872, 20413731]
# SElect the possible outliers
sample1 = x[ids == possible_outliers[0]]
sample2 = x[ids == possible_outliers[1]]

plt.plot(np.mean(masses, axis=0), np.mean(x, axis=0) / 1e4, label="Mean of all sapmles")
plt.plot(
    np.mean(masses, axis=0), np.mean(sample1, axis=0) / 1e4, label="Possible outlier"
)
plt.plot(
    np.mean(masses, axis=0), np.mean(sample2, axis=0) / 1e4, label="Possible outlier"
)
plt.legend()


# Plot possible outliers detected by z-score
# Select the possible outliers
for outlier in outliers[0]:
    plt.figure()
    plt.plot(np.mean(masses, axis=0), np.mean(x, axis=0), label="Mean of all sapmles")
    sample = x[outlier]
    plt.plot(np.mean(masses, axis=0), sample, label="Possible outlier")
    plt.legend()


# # Genearte a pkl with all de data
# paths = ["data/data_exp1.pkl", "data/data_exp3.pkl", "data/data_exp4_brote_gm.pkl", "data/data_exp4_brote_gomez_ulla.pkl"]

# x_total = np.empty((0, 18000))
# ids_total = np.empty((0, 1))
# masses_total = np.empty((0, 18000))
# y_total = np.empty((0, 1))

# for path in paths:
#     with open(main_path + path, "rb") as handle:
#         data = pickle.load(handle)

#     if path == "data/data_exp1.pkl":
#         x = np.vstack((np.vstack(data["train"]["intensities"]), np.vstack(data["test"]["intensities"])))
#         ids = np.vstack((np.vstack(data["train"]["ids"]), np.vstack(data["test"]["ids"])))
#         masses = np.vstack((np.vstack(data["train"]["masses"]), np.vstack(data["test"]["masses"])))
#         y = np.hstack((data["train"]["labels"], data["test"]["labels"]))

#     else:
#         x = np.vstack(data["test"]["intensities"])
#         ids = np.vstack(data["test"]["ids"])
#         masses = np.vstack(data["test"]["masses"])
#         y = data["test"]["labels"]

#     x_total = np.vstack((x_total, x))
#     ids_total = np.vstack((ids_total, ids))
#     masses_total = np.vstack((masses_total, masses))
#     y_total = np.vstack((y_total, y.reshape(-1,1)))

# data = {"intensities": x_total, "ids": ids_total, "masses": masses_total, "labels": y_total}
# # Store data as pickle
# with open(main_path + "data/data_final.pkl", "wb") as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
