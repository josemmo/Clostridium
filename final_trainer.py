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
    maldi_path = main_path + "data/data_final.pkl"
    results = main_path + "results_paper/"

    # ============ Load data ===================
    print("Loading data...")
    with open(maldi_path, "rb") as handle:
        data = pickle.load(handle)

    x = data["intensities"] * 1e4
    masses = data["masses"]
    y = data["labels"]

    # ============ Preprocess data ===================

    # Check if path "results_paper/model" exists, if not, create it
    if not os.path.exists(results + "final_model/" + model + "/"):
        os.makedirs(results + "final_model/" + model + "/")
    results = results + "final_model/" + model

    if model == "base":
        return NotImplementedError

    elif model == "favae":
        from models import FAVAE

        model = FAVAE(latent_dim=20, epochs=100)
        model.fit(x, y)

        pickle.dump(model, open(results + "/model_all.pkl", "wb"))

    elif model == "rf":
        from models import RF

        # Declare the model
        model = RF(max_depth=depth)
        # Train it
        model.fit(x, y)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))

        # Evaluation
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            masses,
            results + "/feature_importance_trainmodel.png",
            wandbflag=wandbflag,
        )

    elif model == "lr":
        from models import LR

        model = LR()
        model.fit(x, y)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))

        # Evaluation
        importances = model.coef_
        for i in range(len(importances)):
            plot_importances(
                model,
                importances[i, :],
                masses,
                results + "/feature_importance_trainmodel_class" + str(i) + ".png",
                wandbflag=wandbflag,
            )
        plot_importances(
            model,
            np.mean(importances, axis=0),
            masses,
            results + "/feature_importance_trainmodel_mean_all_classes.png",
            wandbflag=wandbflag,
        )

    elif model == "dt":
        from models import DecisionTree

        model = DecisionTree(max_depth=depth)
        model.fit(x, y)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))

        # Evaluation
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            masses,
            results + "/feature_importance_trainmodel.png",
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
        choices=["base", "rf", "dt", "favae", "lr"],
    )
    argparse.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    argparse.add_argument("--depth", type=int, default=10, help="Max depth of the tree")
    argparse.add_argument("--wandb", type=bool, default=False, help="Use wandb")

    args = argparse.parse_args()

    main(args.model, args.config, depth=args.depth, wandbflag=args.wandb)

    # python final_trainer.py --model dt --config config.yaml
