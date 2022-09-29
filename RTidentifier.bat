@echo off
set /p Rpath=Insert the path to the R interpreter:
set /p MaldiPath=Insert the path to MALDIs data:
set /p ExportPath=Insert the path where you want to store the results:

echo Preprocessing MALDI using MALDIQuant...

%Rpath% --vanilla preprocess_maldi.R %MaldiPath% %ExportPath%

echo MALDI data preprocessed!

echo Predicting MALDI data...

