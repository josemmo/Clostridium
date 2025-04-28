from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from favae import favae
from dblrfs import DBL_class


class KSSHIBA:
    def __init__(
        self,
        kernel="rbf",
        epochs=10000,
        fs=False,
    ):
        self.kernel = kernel
        self.epochs = epochs
        self.fs = fs
        self.model = favae.SSHIBA(100, 1, fs=self.fs)  # Kc and prune

    def fit(self, x_train, y_train):
        self.x_train = x_train
        maldis = self.model.struct_data(
            self.x_train, method="reg", V=self.x_train, kernel=self.kernel
        )

        # Convert to one hot encoding the labels
        print("Converting to one hot encoding...")
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(y_train.reshape(-1, 1))
        y_train_ohe = ohe.transform(y_train.reshape(-1, 1))
        labels = self.model.struct_data(y_train_ohe, "mult")

        print("Training model...")
        self.model.fit(
            maldis,
            labels,
            max_iter=self.epochs,
            pruning_crit=1e-5,
            verbose=1,
        )
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
        return np.argmax(self.predict_proba(x_test), axis=1)

    def predict_proba(self, x_test):
        maldis_test = self.model.struct_data(
            x_test, method="reg", V=self.x_train, kernel=self.kernel
        )
        y_pred, Z_test_mean, Z_test_cov = self.model.predict([0], [1], maldis_test)
        return y_pred["output_view1"]["mean_x"]

    def get_model(self):
        return self.model


class FAVAE:
    def __init__(
        self,
        latent_dim=10,
        epochs=100,
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.model = favae.SSHIBA(100, 1)  # Kc and prune

    def fit(self, x_train, y_train):
        maldis = self.model.struct_data(
            x_train, "vae", latent_dim=self.latent_dim, lr=1e-3, dataset="clostri"
        )
        print(x_train)

        # Convert to one hot encoding the labels
        print("Converting to one hot encoding...")
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(y_train.reshape(-1, 1))
        y_train_ohe = ohe.transform(y_train.reshape(-1, 1))

        print(y_train_ohe)
        labels = self.model.struct_data(y_train_ohe, "mult")

        print("Training model...")
        self.model.fit(
            maldis,
            labels,
            max_iter=self.epochs,
            pruning_crit=1e-5,
            verbose=1,
        )
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
        return np.argmax(self.predict_proba(x_test), axis=1)

    def predict_proba(self, x_test):
        maldis_test = self.model.struct_data(
            x_test, "vae", latent_dim=self.latent_dim, lr=1e-3, dataset="clostri"
        )
        y_pred, Z_test_mean, Z_test_cov = self.model.predict([0], [1], maldis_test)
        return y_pred["output_view1"]["mean_x"]

    def get_model(self):
        return self.model


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
                "n_estimators": [100],
                "max_depth": [2, 4, 6, 8],
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            scoring="balanced_accuracy",
            cv=self.cv,
            verbose=2,
            n_jobs=-1,
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
            param_grid={
                "max_depth": np.arange(2, self.max_depth, 2),
                "min_samples_split": [2, 4, 6],
                "min_samples_leaf": [1, 2, 3],
                "max_features": ["sqrt", "log2"],
            },
            scoring="balanced_accuracy",
            cv=self.cv,
            verbose=2,
            n_jobs=-1,
        )
        grid_results = grid.fit(x_train, y_train)
        self.model = grid_results.best_estimator_

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


class LR:
    def __init__(self):
        self.cv = 5
        self.model = None

    def fit(self, x_train, y_train):
        ros = RandomOverSampler()
        x_train, y_train = ros.fit_resample(x_train, y_train)
        lr = LogisticRegression(random_state=0, multi_class="multinomial")
        print("Cross-validating using grid search...")
        grid = GridSearchCV(
            estimator=lr,
            param_grid={
                "penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1, 1.0, 10, 100],
                "solver": ["liblinear", "sag"],
            },
            scoring="balanced_accuracy",
            cv=self.cv,
            verbose=2,
            n_jobs=-1,
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


class LR_ARD(object):
    def __init__(self):
        pass

    def fit(self, z, y, z_tst = None, y_tst = None,  hyper = None, maxit = 30,
            pruning = 8e-2):

        self.z = z
        self.t = y
        self.z_tst = z_tst
        self.t_tst = y_tst
        self.pruning = pruning
        self.maxit = maxit

        print('Loading the data...')

        ones_tr = np.ones((np.shape(self.z)[0],1))
        ones_test = np.ones((np.shape(self.z_tst)[0],1))

        x_train = np.hstack((self.z,ones_tr))
        x_test = np.hstack((self.z_tst,ones_test))

        y_train_1 = np.where(self.t==0,1,0)
        y_train_1 = np.reshape(y_train_1,(np.shape(y_train_1)[0],1))
        y_train_2 = np.where(self.t==1,1,0)
        y_train_2 = np.reshape(y_train_2,(np.shape(y_train_2)[0],1))
        y_train_3 = np.where(self.t==2,1,0)
        y_train_3 = np.reshape(y_train_3,(np.shape(y_train_3)[0],1))

        y_tst_1 = np.where(self.t_tst==0,1,0)
        y_tst_1 = np.reshape(y_tst_1,(np.shape(y_tst_1)[0],1))
        y_tst_2 = np.where(self.t_tst==1,1,0)
        y_tst_2 = np.reshape(y_tst_2,(np.shape(y_tst_2)[0],1))
        y_tst_3 = np.where(self.t_tst==2,1,0)
        y_tst_3 = np.reshape(y_tst_3,(np.shape(y_tst_3)[0],1))

        print('Training the models...')

        self.myModel1 = DBL_class.LR_ARD()
        self.myModel1.fit(x_train, y_train_1, x_test, y_tst_1,prune = 0,maxit = self.maxit)
        pesos1 = self.myModel1.return_w()[:-1,:].ravel()
        maximo1 = np.max(np.abs(pesos1))
        self.pesos1 = np.where(np.abs(pesos1)<maximo1*self.pruning,0., pesos1)

        self.myModel2 = DBL_class.LR_ARD()
        self.myModel2.fit(x_train, y_train_2, x_test, y_tst_2,prune = 0, maxit = self.maxit)
        pesos2 = self.myModel2.return_w()[:-1,:].ravel()
        maximo2 = np.max(np.abs(pesos2))
        self.pesos2 = np.where(np.abs(pesos2)<maximo2*self.pruning,0., pesos2)

        self.myModel3 = DBL_class.LR_ARD()
        self.myModel3.fit(x_train, y_train_3, x_test, y_tst_3,prune = 0, maxit = self.maxit)
        pesos3 = self.myModel3.return_w()[:-1,:].ravel()
        maximo3 = np.max(np.abs(pesos3))
        self.pesos3 = np.where(np.abs(pesos3)<maximo3*self.pruning,0., pesos3)

    def predict_proba_true(self, Z_tst):
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs1 = self.myModel1.predict_proba_th(Z_tst,pruning_crit= self.pruning)
        probs2 = self.myModel2.predict_proba_th(Z_tst,pruning_crit= self.pruning)
        probs3 = self.myModel3.predict_proba_th(Z_tst,pruning_crit= self.pruning)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        return probs

    def predict_proba(self, Z_tst):
        #Ojo que esto es cutre cutre

        #Calculamos el minimo y maximo de la prob con las salidas de los datos de train
        probs1 = self.myModel1.predict_proba_th(self.z,pruning_crit= self.pruning)
        probs2 = self.myModel2.predict_proba_th(self.z,pruning_crit= self.pruning)
        probs3 = self.myModel3.predict_proba_th(self.z,pruning_crit= self.pruning)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        #print(probs)

        maximo = np.max(probs.ravel())
        minimo = np.min(probs.ravel())
        #print('Maximo: ', maximo)
        #print('Minimo: ', minimo)
        #Calculamos las probs del test
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs1 = self.myModel1.predict_proba_th(Z_tst,pruning_crit= self.pruning)
        probs2 = self.myModel2.predict_proba_th(Z_tst,pruning_crit= self.pruning)
        probs3 = self.myModel3.predict_proba_th(Z_tst,pruning_crit= self.pruning)

        prob_p = np.hstack((probs1,probs2))
        probs = np.hstack((prob_p,probs3))
        #Normalizamos las probabilidades de salida respecto a los datos de train

        probs_norm = self.normalize_data(probs, maximo, minimo)

        #Chequeamos que ningun casi se ha salido de [0,1]
        probs_norm = np.where(probs_norm > 1.0, 1.0, probs_norm)
        probs_norm = np.where(probs_norm < 0.0, 0.0, probs_norm)
        return probs_norm

    def predict(self, Z_tst):
        ones = np.ones((np.shape(Z_tst)[0],1))
        Z_tst = np.hstack((Z_tst,ones))

        probs = self.predict_proba_true(Z_tst)
        preds = np.argmax(probs, axis = 1)
        return preds

    def return_weights(self):
        return [self.pesos1, self.pesos2, self.pesos3]

    def easter_egg(self):
        print('38.9142,-0.549496')

    def normalize_data(self,X, maximo, minimo):
        return (X - minimo)/(maximo - minimo)
