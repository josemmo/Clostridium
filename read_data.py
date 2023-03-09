def exp1():
    """The folder structure is the following: ./data/Espectros Clostridium Maldi/275 iniciales/.
    Under that folder, there are three folders with the names of the three categories of ribotypes."""

    # Read data and store it in three different dataframes
    df = pd.DataFrame()
    for folder in os.listdir("./data/Espectros Clostridium Maldi/275 iniciales/"):
        if folder == ".DS_Store":
            continue
        for file in os.listdir(
            "./data/Espectros Clostridium Maldi/275 iniciales/" + folder
        ):
            if file == ".DS_Store":
                continue
            df = df.append(
                pd.read_csv(
                    "./data/Espectros Clostridium Maldi/275 iniciales/"
                    + folder
                    + "/"
                    + file,
                    sep="\t",
                    header=None,
                ),
                ignore_index=True,
            )
