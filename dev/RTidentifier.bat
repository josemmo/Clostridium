@echo off

set /p Replicates=Has the MALDI any replicates?:

echo Preprocessing MALDI using MALDIQuant...

Rscript .\preprocess_maldi.R %Replicates%


echo MALDI data preprocessed!

echo Predicting MALDI data...

