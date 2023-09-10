import argparse
from sklearn.model_selection import train_test_split
import yaml
import pickle
import numpy as np
import os
import pandas as pd


def preprocess_data(data_path, store_preprocess_data):
    # Load data

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
            intensities.append(aux[:18000, 1])
            # The sample id is the name of the csv file
            sample_ids.append(file.split(".")[0])

    # Convert to numpy array
    masses = np.array(masses)
    intensities = np.array(intensities) * 1e4
    print("Data loaded")
    return masses, intensities, sample_ids


def predict(
    models, data_path, intensities, sample_ids, labels, medios, semanas, grupos
):
    columns = ["Sample"]
    for model_name in models:
        columns.append(model_name)
        columns.append(model_name + " probability")
    columns.append("Medio")
    columns.append("True_label")
    columns.append("Semana")
    columns.append("Grupo")

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

        # Map labels to RT027, RT181, Others
        labels = np.array(labels)

        labels = np.where(labels == "027", "RT027", labels)
        labels = np.where(labels == "181", "RT181", labels)
        labels = np.where((labels != "RT027") & (labels != "RT181"), "Others", labels)

        # Store in results dataframe a new row for each sample with the prediction of each model
        for i in range(len(y_pred)):
            results["Sample"][i] = sample_ids[i]
            results[model_name][i] = y_pred[i]
            results[model_name + " probability"][i] = np.max(y_pred_proba[i])
            results["Medio"][i] = medios[i]
            results["True_label"][i] = labels[i]
            results["Semana"][i] = semanas[i]
            results["Grupo"][i] = grupos[i]

    # Calculate accurary
    total_acc = (results["DBLFS"] == results["True_label"]).sum() / len(results)
    # Accuracy per medio
    medio_acc = [results["Medio"].unique()]
    for medio in results["Medio"].unique():
        medio_acc.append(
            (
                results[results["Medio"] == medio]["DBLFS"]
                == results[results["Medio"] == medio]["True_label"]
            ).sum()
            / len(results[results["Medio"] == medio])
        )

    # Accuracy per sample
    sample_acc = [results["Sample"].unique()]
    for sample in results["Sample"].unique():
        sample_acc.append(
            (
                results[results["Sample"] == sample]["DBLFS"]
                == results[results["Sample"] == sample]["True_label"]
            ).sum()
            / len(results[results["Sample"] == sample])
        )
    # Accuracy por semana
    semana_acc = [results["Semana"].unique()]
    for semana in results["Semana"].unique():
        semana_acc.append(
            (
                results[results["Semana"] == semana]["DBLFS"]
                == results[results["Semana"] == semana]["True_label"]
            ).sum()
            / len(results[results["Semana"] == semana])
        )

    # Accuracy por grupo
    grupo_acc = [results["Grupo"].unique()]
    for grupo in results["Grupo"].unique():
        grupo_acc.append(
            (
                results[results["Grupo"] == grupo]["DBLFS"]
                == results[results["Grupo"] == grupo]["True_label"]
            ).sum()
            / len(results[results["Grupo"] == grupo])
        )

    print("Total accuracy: " + str(total_acc))
    print("Accuracy per medio: " + str(medio_acc))
    print("Accuracy per sample: " + str(sample_acc))
    print("Accuracy per semana: " + str(semana_acc))
    print("Accuracy per grupo: " + str(grupo_acc))

    print("Storing results in csv...")

    # Store results dataframe in csv
    results.to_csv(path_to_results + "results.csv", index=False)


def read_repro(data_path):
    print("Loading data preprocessed (masses, intensities, sample_ids)")
    masses = []
    intensities = []
    sample_ids = []
    labels = []
    medios = []
    semanas = []
    grupos = []

    for semana in os.listdir(data_path):
        for grupo in os.listdir(data_path + "/" + semana):
            for medio in os.listdir(data_path + "/" + semana + "/" + grupo):
                for file in os.listdir(
                    data_path + "/" + semana + "/" + grupo + "/" + medio
                ):
                    aux = np.loadtxt(
                        data_path
                        + "/"
                        + semana
                        + "/"
                        + grupo
                        + "/"
                        + medio
                        + "/"
                        + file,
                        delimiter=",",
                        skiprows=1,
                    )

                    masses.append(aux[:18000, 0])
                    intensity = aux[:18000, 1]
                    # Normalise intensity by Total Ion Current method (all intensity have to sum up to 1)
                    intensity = intensity / np.sum(intensity)
                    intensities.append(intensity)
                    # The sample id is the name of the csv file
                    sample_ids.append(file.split("_")[0])
                    medios.append(medio.split(" ")[1])
                    labels.append(file.split("_")[1])
                    semanas.append(semana.split(" ")[0])
                    grupos.append(grupo.split(" ")[1])

    # Convert to numpy array
    masses = np.array(masses)
    intensities = np.array(intensities) * 1e4
    print("Data loaded")
    return masses, intensities, sample_ids, labels, medios, semanas, grupos


def main(data_path, model_name):
    data_path = "/export/usuarios01/alexjorguer/Datos/HospitalProject/Clostridium/repro/Estudio Reproducibilidad (MALDI)"
    # Preprocess data using R script
    masses, intensities, sample_ids, labels, medios, semanas, grupos = read_repro(
        data_path
    )

    # Define models to use
    if model_name is None:
        models = ["DT", "RF", "DBLFS", "LR"]
    else:
        models = [model_name]

    # Predict
    predict(models, data_path, intensities, sample_ids, labels, medios, semanas, grupos)


if __name__ == "__main__":
    # argparse = argparse.ArgumentParser()
    # argparse.add_argument("--data", type=str, default="rf", help="Path to the data")

    # argparse.add_argument(
    #     "--model",
    #     type=str,
    #     default=None,
    #     help="Model to use, if None, all models are used to predict",
    # )

    # args = argparse.parse_args()

    # main(args.data, args.model)

    main("user_A/Gomez_Ulla", "DBLFS")

    # python predict.py --data user_A/Gomez_Ulla --model rf
