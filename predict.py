import argparse
from sklearn.model_selection import train_test_split
import yaml
import pickle
import numpy as np
import os
import pandas as pd


def preprocess_data(data_path, store_preprocess_data):
    print("Preprocessing data with R script...")
    os.system("Rscript preprocess_maldi.R " + data_path + " " + store_preprocess_data)
    print("Data preprocessed")

    print("Loading data preprocessed (masses, intensities, sample_ids)")
    masses = []
    intensities = []
    sample_ids = []
    for folder in os.listdir(store_preprocess_data):
        for file in os.listdir(store_preprocess_data + folder):
            aux = np.loadtxt(
                store_preprocess_data + folder + "/" + file, delimiter=",", skiprows=1
            )

            masses.append(aux[:18000, 0])
            intensities.append(aux[:18000, 1] * 1e4)
            # The sample id is the name of the csv file
            sample_ids.append(file.split(".")[0])

    # Convert to numpy array
    masses = np.array(masses)
    intensities = np.array(intensities)
    print("Data loaded")
    return masses, intensities, sample_ids


def predict(models, data_path, intensities, sample_ids):
    columns = ["Sample"]
    for model_name in models:
        columns.append(model_name)
        columns.append(model_name + " probability")
    # Generate a dataframe witht he columns columns=["Sample", "DT", "DT probability", "RF", "RF probability", "FAVAE", "FAVAE probability"] and rows equal to number of samples
    results = pd.DataFrame(columns=columns, index=range(len(intensities)))
    for model_name in models:
        print("Predicting with model: " + model_name)
        path_to_results = data_path + "/results/"
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)

        with open(
            "results_paper/final_model/" + model_name.lower() + "/model_all.pkl", "rb"
        ) as handle:
            model = pickle.load(handle)

        # Predict
        y_pred = np.array(np.array(model.predict(intensities), dtype=int), dtype=str)
        y_pred_proba = model.predict_proba(intensities)

        if model_name == "FAVAE":
            y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

        # Map y_pred: where y_pred is 0, map to RT027, where y_pred is 1, map to RT181, otherwise map to Others
        y_pred = np.where(y_pred == "0", "RT027", y_pred)
        y_pred = np.where(y_pred == "1", "RT181", y_pred)
        y_pred = np.where(y_pred == "2", "Others", y_pred)

        # Store in results dataframe a new row for each sample with the prediction of each model
        for i in range(len(y_pred)):
            results["Sample"][i] = sample_ids[i]
            results[model_name][i] = y_pred[i]
            results[model_name + " probability"][i] = np.max(y_pred_proba[i])

    print("Storing results in csv...")

    # Store results dataframe in csv
    results.to_csv(path_to_results + "results.csv", index=False)


def main(data_path):
    # Preprocess data using R script
    store_preprocess_data = data_path + "/results/data_processed/"
    # Check if store_preprocess_data exists
    if not os.path.exists(store_preprocess_data):
        os.makedirs(store_preprocess_data)
    masses, intensities, sample_ids = preprocess_data(data_path, store_preprocess_data)

    # Define models to use
    models = ["DT", "RF"]

    # Predict
    predict(models, data_path, intensities, sample_ids)

    # Remove data preprocessed
    os.system("rm -r " + store_preprocess_data)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data", type=str, default="rf", help="Path to the data")
    args = argparse.parse_args()

    main(args.data)

    # python predict.py --data user_A/Gomez_Ulla
