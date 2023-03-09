import os
import numpy as np
import pandas as pd

# Read all folders tree that are under /data/all_maldi/
path = "./data/all_maldi/initial/027"
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

# Read an excel
errores = pd.read_excel(
    "/export/usuarios01/alexjorguer/Datos/HospitalProject/Clostridium/data/guiacepas.xlsx"
)

# Get the name of the strain which is the 5th element of the path
strain_path = [listOfFiles[i].split("/")[:6] for i in range(len(listOfFiles))]
# Remove duplicates from strain_path without using tuples
strain_path = [list(x) for x in set(tuple(x) for x in strain_path)]

for i in strain_path:
    full_strain_name = i[5]
    old_path = i.copy()
    # split by whitespaces and get the last item
    strain_name = full_strain_name.split(" ")[-1]
    if len(strain_name.split("-")[0]) > 3:
        incorrect_name = reversed(strain_name.split("-"))
        correct_name = "-".join(incorrect_name)
        i[5] = correct_name
        # Rename the folder

        # Remove old path from strain_path list
    else:
        i[5] = strain_name

    name = i[5].split("-")[1]
    # Find which cell from errores has the correct name
    correct_name = errores[errores["N cepa"] == name]["correct_name"].values[0]

    os.rename("/".join(old_path), "/".join(i))


# join them
strain_path = ["/".join(strain_path[i]) for i in range(len(strain_path))]
