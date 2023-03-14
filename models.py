from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time
import numpy as np


class RF:
    def __init__(self, n_estimators=128, max_depth=10, cv=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv = 5
        self.model = None

    def fit(self, x_train, y_train):
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)
        dfrst = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )
        print("Cross-validating using grid search...")
        grid = GridSearchCV(
            estimator=dfrst,
            param_grid={
                "max_depth": np.arange(2, self.max_depth, 2),
                "n_estimators": np.arange(64, self.n_estimators, 64),
            },
            scoring="balanced_accuracy",
            cv=self.cv,
            verbose=2,
        )
        grid_results = grid.fit(x_train, y_train)
        self.model = grid_results.best_estimator_

        return self.model

    def save(self, path):
        model = {"model": self.model}
        with open(path, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        self.model = model["model"]
        return self.model

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_model(self):
        return self.model


class DecisionTree:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.cv = 5
        self.model = None

    def fit(self, x_train, y_train):
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(random_state=0)
        print("Cross-validating using grid search...")
        grid = GridSearchCV(
            estimator=clf,
            param_grid={"max_depth": np.arange(2, self.max_depth, 2)},
            scoring="balanced_accuracy",
            cv=self.cv,
            verbose=2,
        )
        grid_results = grid.fit(x_train, y_train)
        self.model = grid_results.best_estimator_

    def save(self, path):
        model = {"model": self.model}
        with open(path, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        self.model = model["model"]
        return self.model

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_model(self):
        return self.model
