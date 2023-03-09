import pandas as pd
import os
import pickle
import numpy as np


def read_train_data(path):
    # os.walk all the files in the data folder:
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # Read labels from todas_labels.xlsx
    labels = pd.read_excel("data/todas_labels.xlsx", header=None)
    ids = [int(file.split("/")[-1].split("_")[0]) for file in listOfFiles]


    # Read each csv of listOfFiles and append it to df to the MALDI column
    data = []
    for file in listOfFiles:
        d = pd.read_csv(file)
        id = int(file.split("/")[-1].split("_")[0])
        data.append(
            [
                d["mass"].values[:18000],
                d["intensity"].values[:18000],
                str(labels[labels[1]==id][2].values[0]),
            ]
        )
    df = pd.DataFrame(data, columns=["MALDI_mass", "MALDI_int", "label"])

    # Substitute the label with the number of the class: '027' -> 0, '181' -> 1, rest -> 2
    df["label"] = df["label"].replace({"027": 0, "181": 1})
    # All other labels are 2
    df["label"] = df["label"].replace(
        {"001": 2, "002": 2, "014": 2, "017": 2, "023": 2, "078": 2, "106": 2, "207": 2}
    )

    # TODO: ME HE QUEDADO AQUÍ, FALTA CLASIFICAR EL RT 165 COMO 2

    # Store a pickle with a dictionary with the data
    maldis = np.vstack(df["MALDI_int"].values)
    masses = np.vstack(df["MALDI_mass"].values)
    labels = df["label"].values

    data = {"maldis": maldis, "masses": masses, "labels": labels}
    # Save the data
    with open("data/data_processed_noreplicas_090502023.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_test_data(path):

