# Clostridium difficile ribotype classifier
by
Alejandro Guerrero-López

Preliminary ideas and experiments on Clostridium difficile ribotype classificaiton.

## Dependencies

### Python
You'll need a working Python environment to run the code.
The recommended way is by using
[Anaconda Python distribution](https://www.anaconda.com/download/).
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation. Hence, you can install our dependencies without causing conflicts with your
setup.

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create -f environment_win11.yml
    conda activate clostri

### R MALDIquant preprocessing step
Install R-4.1.1. All tests were tried under this specific version. Moreover, you will need to have the following R libraries:

            install.packages(c("MALDIquant","MALDIquantForeign", "stringr"))

## How to use the script

Create the following folder:

        mkdir ./data_to_predict

Then, store in it the MALDI raw data as it comes from Bruker's MALDI machine, i.e., following the next structure IF THE SAMPLE HAS NO REPLICATES:

 ```
./data_to_predict
│
└───ID_of_the_first_MALDI-TOF (for example 22317388)
│   │
│   └───Position_of_the_sample (for example 0_C1)
│       └───1
|           └───1SLin
|               └───pdata
|               |   acqu
|               |   acqu.org
|               |   acqus
|               |   acqus.org
|               |   fid
|               |   sptype  
└───ID_of_the_second_MALDI-TOF (for example 22319345)
│   │
│   └───Position_of_the_sample (for example 0_C5)
│       └───1
|           └───1SLin
|               └───pdata
|               |   acqu
|               |   acqu.org
|               |   acqus
|               |   acqus.org
|               |   fid
|               |   sptype     
│   
└───...
```
If the MALDI has replicates, store it as follows:
 ```
./data_to_predict
│
└───ID_of_the_first_MALDI-TOF (for example 22317388)
|    └───Folder of replica 1 (for example D1)
│    |   └───Position_of_the_sample (for example 0_C1)
│    |       └───1
|    |           └───1SLin
|    |               └───pdata
|    |               |   acqu
|    |               |   acqu.org
|    |               |   acqus
|    |               |   acqus.org
|    |               |   fid
|    |               |   sptype  
|    └───Folder of replica 2 (for example D2)
│       └───Position_of_the_sample (for example 0_C5)
│           └───1
|               └───1SLin
|                   └───pdata
|                   |   acqu
|                   |   acqu.org
|                   |   acqus
|                   |   acqus.org
|                   |   fid
|                   |   sptype  
└───...
```

The predictions will be stored in results as an .xlsx file. Then, for each MALDI-TOF, an image will be created in `./results/images` representing the 6 different ROIs.