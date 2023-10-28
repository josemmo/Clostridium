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
    # maldi_data_path = main_path + "data/data_exp1.pkl"
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
    maldi_train_path = "data/df_train_exp2.pkl"
    maldi_test_path = "data/df_test_exp2.pkl"
    with open(maldi_train_path, "rb") as handle:
        data_train = pickle.load(handle)
    with open(maldi_test_path, "rb") as handle:
        data_test = pickle.load(handle)

    x_train = np.array(data_train["intensity"].tolist())
    y_train = np.array(data_train["label"].tolist())
    x_test = np.array(data_test["intensity"].tolist())
    y_test = np.array(data_test["label"].tolist())
    
    # Convert the labels: if 027 is 0, if 181 is 1, all the rest is 2
    y_train[y_train == "027"] = 0
    y_train[y_train == "181"] = 1
    y_train[y_train == "other"] = 2
    y_test[y_test == "027"] = 0
    y_test[y_test == "181"] = 1
    y_test[y_test == "other"] = 2
    

    # ============ Preprocess data ===================

    if wandbflag:
        # Save number of samples in the dataset
        wandb.log({"Number of samples": len(np.vstack((x_train, x_test)))})
        # Save number of features in the dataset
        wandb.log({"Number of features": len(np.vstack((x_train, x_test)))})
        # Save number of samples in train
        wandb.log({"Number of samples in train": len(x_train)})
        # Save number of samples in test
        wandb.log({"Number of samples in test": len(x_test)})

    # Check if path "results_paper/model" exists, if not, create it
    if not os.path.exists(results + "exp1/" + model + "/"):
        os.makedirs(results + "exp1/" + model + "/")
    results = results + "exp1/" + model

    if model == "base":
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(x_train, x_test, y_train, y_test)

        print(models)

    elif model == "ksshiba":
        from models import KSSHIBA

        kernel = "linear"
        epochs = 1000
        fs = True

        results = (
            results + "_kernel_" + kernel + "_epochs_" + str(epochs) + "_fs_" + str(fs)
        )
        # Check if path "results_paper/model" exists, if not, create it
        if not os.path.exists(results):
            os.makedirs(results)

        print("Training KSSHIBA")
        model = KSSHIBA(kernel=kernel, epochs=epochs, fs=fs)

        # model.fit(x_train, y_train)

        # print("Evaluating KSSHIBA")
        # # # Evaluation
        # pred = model.predict(x_test)
        # pred_proba = model.predict_proba(x_test)
        # pred_proba = pred_proba / pred_proba.sum(axis=1)[:, None]

        # multi_class_evaluation(
        #     y_test,
        #     pred,
        #     pred_proba,
        #     results_path=results,
        #     wandbflag=wandbflag,
        # )

        model.fit(np.vstack((x_train, x_test)), np.hstack((y_train, y_test)))
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))

    elif model == "favae":
        from models import FAVAE

        model = FAVAE(latent_dim=100, epochs=1000)
        model.fit(x_train, y_train)

        # # Evaluation
        pred = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)
        pred_proba = pred_proba / pred_proba.sum(axis=1)[:, None]

        multi_class_evaluation(
            y_test,
            pred,
            pred_proba,
            results_path=results,
            wandbflag=wandbflag,
        )

        model.fit(np.vstack((x_train, x_test)), np.hstack((y_train, y_test)))
        store = {"model": model, "x_train": x_train, "y_train": y_train}
        pickle.dump(store, open(results + "/model_all.pkl", "wb"))

    elif model == "rf":
        from models import RF

        # Declare the model
        model = RF(max_depth=depth)
        # Train it
        model.fit(x_train, y_train)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model.pkl", "wb"))

        # Evaluation
        if wandbflag:
            wandb.sklearn.plot_learning_curve(model, x_train, y_train)

        # Evaluation
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            x_train_masses,
            results + "/feature_importance_trainmodel.png",
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

        # Retrain the model with all data and save it
        model = RF(max_depth=depth)
        model.fit(np.vstack((x_train, x_test)), np.hstack((y_train, y_test)))
        model = model.get_model()
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            x_total_masses,
            results + "/feature_importance_completemodel.png",
            wandbflag=wandbflag,
        )

    elif model == "dblfs":
        from models import DBLFS

        # Declare the model
        pred_proba = []
        for i in np.arange(y_train.shape[1]):
            model = DBLFS()
            # TODO: y_train must be ohe, check it
            model.fit(x_train, y_train[:, i])
            pred_proba_i = model.predict_proba(x_test)
            pred_proba.append(pred_proba_i)
        pred_proba = np.array(pred_proba).T
        pred = np.argmax(pred_proba, axis=1)

    elif model == "lr":
        from models import LR

        model = LR()
        model.fit(x_train, y_train)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model.pkl", "wb"))

        if wandbflag:
            wandb.sklearn.plot_learning_curve(model, x_train, y_train)

        # Evaluation
        importances = model.coef_
        for i in range(len(importances)):
            plot_importances(
                model,
                importances[i, :],
                x_train_masses,
                results + "/feature_importance_trainmodel_class" + str(i) + ".png",
                wandbflag=wandbflag,
            )
        plot_importances(
            model,
            np.mean(importances, axis=0),
            x_train_masses,
            results + "/feature_importance_trainmodel_mean_all_classes.png",
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
        model = LR()
        model.fit(np.vstack((x_train, x_test)), np.hstack((y_train, y_test)))
        model = model.get_model()
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))
        importances = model.coef_
        for i in range(len(importances)):
            plot_importances(
                model,
                importances[i, :],
                x_train_masses,
                results + "/feature_importance_completemodel_class" + str(i) + ".png",
                wandbflag=wandbflag,
            )
        plot_importances(
            model,
            np.mean(importances, axis=0),
            x_train_masses,
            results + "/feature_importance_completemodel_mean_all_classes.png",
            wandbflag=wandbflag,
        )

    elif model == "dt":
        from models import DecisionTree

        model = DecisionTree(max_depth=depth)
        model.fit(x_train, y_train)
        model = model.get_model()

        # save the model to disk
        pickle.dump(model, open(results + "/model.pkl", "wb"))

        if wandbflag:
            wandb.sklearn.plot_learning_curve(model, x_train, y_train)

        # Evaluation
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            x_train_masses,
            results + "/feature_importance_trainmodel.png",
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
        model = DecisionTree(max_depth=depth)
        model.fit(np.vstack((x_train, x_test)), np.hstack((y_train, y_test)))
        model = model.get_model()
        pickle.dump(model, open(results + "/model_all.pkl", "wb"))
        importances = model.feature_importances_
        plot_importances(
            model,
            importances,
            x_total_masses,
            results + "/feature_importance_completemodel.png",
            wandbflag=wandbflag,
        )
        # print("Plotting final tree...")
        # plot_tree(
        #     model,
        #     np.vstack((x_train, x_test)),
        #     np.hstack((y_train, y_test)),
        #     x_total_masses,
        #     results + "/complete_tree.svg",
        #     wandbflag=wandbflag,
        # )

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
        choices=["base", "rf", "dt", "favae", "lr", "ksshiba"],
    )
    argparse.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    argparse.add_argument("--depth", type=int, default=10, help="Max depth of the tree")
    argparse.add_argument("--wandb", type=bool, default=False, help="Use wandb")

    args = argparse.parse_args()

    main(args.model, args.config, depth=args.depth, wandbflag=args.wandb)

    # python main_trainer.py --model rf --config config.yaml
