#!/bin/bash
echo Insert the path to the R interpreter:
read Rpath

echo Insert the path to MALDIs data:
read MaldiPath

mkdir ./results

echo Preprocessing MALDI using MALDIQuant...

$Rpath --vanilla preprocess_maldi.R $MaldiPath results/data_processed

echo MALDI data preprocessed!


echo Insert the path to the Python interpreter:
read Pypath

$Pypath predictRT.py --maldi_path results/data_preprocessed