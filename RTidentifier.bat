@echo off

set /p datalocated=Are the MALDI-TOFs to predict located at ./data_to_predict? Insert 1 for yes and 0 for no

set /p Replicates=Has the MALDI any replicates? Insert 1 for yes and 0 for no:

if not exist ".\results" mkdir .\results
if not exist ".\results\images" mkdir .\results\images
if not exist ".\results\data_maldiquant" mkdir .\results\data_maldiquant

echo Preprocessing MALDI using MALDIQuant...

Rscript .\preprocess_maldi.R %Replicates%

echo MALDI data preprocessed!

echo Predicting Ribotypes using python models...

python predictRT.py

echo Prediction done, see you.