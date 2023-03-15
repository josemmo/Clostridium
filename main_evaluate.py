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


def main(model, config, depth=None, wandbflag=False):

    # ============ Load config ===================
    print("Loading config")
    with open(
        "/export/usuarios01/alexjorguer/Datos/HospitalProject/Clostridium/" + config
    ) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    main_path = config["main_path"]
    maldi_data_path = main_path + "data/data_exp3.pkl"
    results = main_path + "results_paper/"

    # ============ Wandb ===================
    if wandbflag:
        config_dict = {
            "static_config": config,
            "hyerparams": {"depth": depth, "model": model},
        }
        wandb.init(
            project="clostridium",
            entity="alexjorguer",
            group="AutoCRT",
            config=config_dict,
        )

    # ============ Load data ===================
    print("Loading data...")
    with open(maldi_data_path, "rb") as handle:
        data = pickle.load(handle)
    print(data.keys())
    x_test = np.vstack(data["test"]["intensities"]) * 1e4
    y_test = data["test"]["labels"]
    x_masses = np.vstack(np.array(data["test"]["masses"]))

    # Check if path "results_paper/model" exists, if not, create it
    if not os.path.exists(results + "exp3/" + model + "/"):
        os.makedirs(results + "exp3/" + model + "/")
    results = results + "exp3/" + model + "/"

    if model == "base":
        raise ValueError("Base model not implemented yet")

    if model == "rf":
        # Load results from experiment 1 from a pkl
        with open(main_path + "results_paper/exp1/rf/metrics.pkl", "rb") as handle:
            metrics = pickle.load(handle)
        print("Results in experiment 1:")
        for key in metrics.keys():
            print(key)
            print(metrics[key])

        # Load model from pickle file
        with open(main_path + "results_paper/exp1/rf/model_all.pkl", "rb") as handle:
            model = pickle.load(handle)

        # Evaluation
        plot_importances(model, x_masses, results, wandbflag=wandbflag)
        pred = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)

        multi_class_evaluation(
            y_test,
            pred,
            pred_proba,
            results_path=results,
            wandbflag=wandbflag,
        )

    elif model == "dt":
        # Load results from experiment 1 from a pkl
        with open(main_path + "results_paper/exp1/rf/metrics.pkl", "rb") as handle:
            metrics = pickle.load(handle)
        print("Results in experiment 1:")
        for key in metrics.keys():
            print(key)
            print(metrics[key])

        # Load model from pickle file
        with open(main_path + "results_paper/exp1/dt/model_all.pkl", "rb") as handle:
            model = pickle.load(handle)

        if wandbflag:
            wandb.sklearn.plot_learning_curve(model, x_test, y_test)

        # Evaluation
        plot_importances(model, x_masses, results, wandbflag=wandbflag)

        pred = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)

        print("Results in experiment 3:")
        multi_class_evaluation(
            y_test,
            pred,
            pred_proba,
            results_path=results,
            wandbflag=wandbflag,
        )

    elif model == "favae":
        raise ValueError("Model not implemented")
    else:
        raise ValueError("Model not implemented")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--model",
        type=str,
        default="base",
        help="Model to train",
        choices=["base", "rf", "dt", "favae"],
    )
    argparse.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    argparse.add_argument("--depth", type=int, default=10, help="Max depth of the tree")
    argparse.add_argument("--wandb", type=bool, default=False, help="Use wandb")

    args = argparse.parse_args()

    main(args.model, args.config, depth=args.depth, wandbflag=args.wandb)
