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
    maldi_data_path = main_path + "data/data_processed_noreplicas_090502023.pkl"
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

    maldis_original = data["maldis"] * 1e4
    y3 = data["labels"]
    masses_original = np.array(data["masses"])

    # ============ Preprocess data ===================
    # Split train and test
    print("Splitting train and test...")
    x_train, x_test, y_train, y_test = train_test_split(
        maldis_original,
        y3,
        train_size=0.7,
    )
    if wandbflag:
        # Save number of samples in the dataset
        wandb.log({"Number of samples": len(maldis_original)})
        # Save number of features in the dataset
        wandb.log({"Number of features": len(maldis_original[0])})
        # Save number of samples in train
        wandb.log({"Number of samples in train": len(x_train)})
        # Save number of samples in test
        wandb.log({"Number of samples in test": len(x_test)})

    # Check if path "results_paper/model" exists, if not, create it
    if not os.path.exists(results + model):
        os.makedirs(results + model)
        results = results + model + "/"

    if model == "base":
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(x_train, x_test, y_train, y_test)

        print(models)

    if model == "rf":
        from models import RF

        # Declare the model
        model = RF()
        # Train it
        model.fit(x_train, y_train)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "model.pkl", "wb"))

        # Evaluation
        plot_importances(model, masses_original, results, wandbflag=wandbflag)
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
        from models import DecisionTree

        model = DecisionTree(max_depth=depth)
        model.fit(x_train, y_train)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "model.pkl", "wb"))

        if wandbflag:
            wandb.sklearn.plot_learning_curve(model, x_train, y_train)

        # Evaluation
        plot_importances(model, masses_original, results, wandbflag=wandbflag)

        plot_tree(
            model,
            x_train,
            y_train,
            masses_original,
            results + "train_tree",
            wandbflag=wandbflag,
        )

        pred = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)

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
