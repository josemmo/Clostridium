import argparse
from sklearn.model_selection import train_test_split
import yaml
from imblearn.over_sampling import RandomOverSampler
import pickle
import numpy as np
from performance_tools import plot_tree, plot_importances, multi_class_evaluation
import wandb
from lazypredict.Supervised import LazyClassifier
import os


def main(model, config, data):
    # Preprocess data using R script
    store_preprocess_data = data + "/results/data_processed/"
    os.system("Rscript preprocess_maldi.R " + data + " " + store_preprocess_data)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "model",
        type=str,
        default="base",
        help="Model to train",
        choices=["base", "rf", "dt", "favae", "lr"],
    )
    argparse.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    argparse.add_argument("--data", type=str, default="rf", help="Path to the data")
    args = argparse.parse_args()

    main(args.model, args.config, args.data)

    # python final_trainer.py --model rf --config config.yaml
