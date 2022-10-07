# Clostridium difficile ribotype classifier
by
Alejandro Guerrero-López

Preliminary ideas and experiments on Clostridium difficile ribotype classificaiton.

## To run this in Windows
1. Install Miniconda

    - Create a conda env using environment_win11.yml
2. Install R-4.1.1

    - Install R libraries:

            install.packages(c("MALDIquant","MALDIquantForeign", "stringr"))

3. Create the following folder tree:

        mkdir ./results
        mkdir ./data_to_predict
        mkdir ./models

    The MALDI raw data has to be placed at `data_to-predict` folder.

    The predictions will be stored in results as an .xlsx file.

    For each MALDI-TOF, an image will be created in `./results/images` representing the 6 different ROIs.