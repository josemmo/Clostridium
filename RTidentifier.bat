@echo off

set /p datalocated=Are the MALDI-TOFs to predict located at ./data_to_predict?

set /p Replicates=Has the MALDI any replicates?:

echo Preprocessing MALDI using MALDIQuant...

Rscript .\preprocess_maldi.R %Replicates%

echo MALDI data preprocessed!


echo Predicting Ribotypes using python models...

python predictRT.py

echo Prediction done, see you.