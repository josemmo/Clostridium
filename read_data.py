import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(path, rawpath, data="train"):
    # os.walk all the files in the data folder:
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    listOfFilesraw = list()
    for (dirpath, dirnames, filenames) in os.walk(rawpath):
        listOfFilesraw += [os.path.join(dirpath, file) for file in filenames]

    # For each path in listoffilesraw only keep the first 7 folders of the path
    listOfFilesraw = list(
        set(["/".join(file.split("/")[:12]) for file in listOfFilesraw])
    )

    # Read labels from todas_labels.xlsx
    labels = pd.read_excel("data/todas_labels.xlsx", header=None)
    ids = [int(file.split("/")[-1].split("_")[0]) for file in listOfFiles]

    # Read each csv of listOfFiles and append it to df to the MALDI column
    if data == "test":
        data = []
        for file in listOfFiles:
            d = pd.read_csv(file)
            id = int(file.split("/")[-1].split("_")[0])
            data.append(
                [
                    id,
                    d["mass"].values[:18000],
                    d["intensity"].values[:18000],
                    str(labels[labels[1] == id][2].values[0]),
                ]
            )
        df = pd.DataFrame(data, columns=["id", "MALDI_mass", "MALDI_int", "label"])
    elif data == "train":
        data = []
        for file in listOfFiles:
            d = pd.read_csv(file)
            id = int(file.split("/")[-2].split("-")[1])
            label = file.split("/")[-2].split("-")[0]
            data.append(
                [
                    id,
                    d["mass"].values[:18000],
                    d["intensity"].values[:18000],
                    str(label),
                ]
            )
        df = pd.DataFrame(data, columns=["id", "MALDI_mass", "MALDI_int", "label"])

    # Substitute the label with the number of the class: '027' -> 0, '181' -> 1, rest -> 2
    df["label"] = df["label"].replace({"027": 0, "181": 1})
    # All other labels are 2
    for index in df["label"].value_counts().index:
        if index != 0 and index != 1:
            df["label"] = df["label"].replace({index: 2})

    # Split train and test by "id" column
    if data == "train":
        ids_to_split = df["id"].unique()
        df_train = df[df["id"].isin(ids_to_split[: int(len(ids_to_split) * 0.7)])]
        df_test = df[df["id"].isin(ids_to_split[int(len(ids_to_split) * 0.7) :])]
        # check that the any id in train is not in test
        assert len(set(df_train["id"].unique()) & set(df_test["id"].unique())) == 0
        # Store them as xlsx
        df_train.to_excel("data/train_exp1.xlsx", index=False)
        df_test.to_excel("data/val_exp1.xlsx", index=False)
        # Train dictionary
        train = {
            "ids": df_train["id"].values,
            "masses": df_train["MALDI_mass"].values,
            "intensities": df_train["MALDI_int"].values,
            "labels": df_train["label"].values,
        }
        # Test dictionary
        test = {
            "ids": df_test["id"].values,
            "masses": df_test["MALDI_mass"].values,
            "intensities": df_test["MALDI_int"].values,
            "labels": df_test["label"].values,
        }
        data = {"train": train, "test": test}
        with open("data/data_exp1.pkl", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Given the ids in train and test, create two different folders with the raw data splitted
        for file in listOfFilesraw:
            id = int(file.split("/")[-1].split("-")[1])
            if id in df_train["id"].values:
                os.system(f"cp -R {file} data/exp1/rain")
            elif id in df_test["id"].values:
                os.system(f"cp -R {file} data/exp1/val")

    elif data == "test":
        # Store it as xlsx valled test exp3
        df.to_excel("data/test_exp3.xlsx", index=False)
        # Test dictionary
        test = {
            "ids": df["id"].values,
            "masses": df["MALDI_mass"].values,
            "intensities": df["MALDI_int"].values,
            "labels": df["label"].values,
        }
        data = {"test": test}
        with open("data/data_exp3.pkl", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
