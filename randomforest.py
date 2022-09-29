from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
import numpy as np
import pickle

with open('data/MALDI/data_processedMALDIQuant_noalign.pickle', 'rb') as handle:
    data = pickle.load(handle)

maldis = data['maldis']
y3 = data['labels_3cat']
masses = data['masses']

results = {}
c=0

ros = RandomOverSampler()
maldis_resampled, y_resampled = ros.fit_resample(maldis, y3)

x_train, x_test, y_train, y_test = train_test_split(maldis_resampled, y_resampled, train_size=0.6)

max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train, y_train)

best_clf = grid_results.best_estimator_
y_pred_maldi = best_clf.predict(x_test)

acc_maldi = balanced_accuracy_score(y_test, y_pred_maldi)
print(acc_maldi)

plt.figure()
plt.title("Using only MALDI")
plt.plot(np.mean(x_test, axis=0), label='Signal mean in test')
plt.plot(grid.best_estimator_.feature_importances_, label='Feature importances', alpha=0.2)
plt.legend()
plt.show()

masses = data['masses']
meanmas = np.mean(masses, axis=0)
imp_peaks_loc = np.where(grid.best_estimator_.feature_importances_>10*np.std(grid.best_estimator_.feature_importances_))[0]
imp_peaks_masses = meanmas[imp_peaks_loc]


p027 = np.where(data['labels_3cat']==0)[0]
p181 = np.where(data['labels_3cat']==1)[0]
pxxx = np.where(data['labels_3cat']==2)[0]

rois = [[2450,2500], [3300,3380], [4700,4750], [4850,4900],  [4900,4950], [4950,5000], [5000,5070], [6600, 6700], [6700,6750]]
fig, ax = plt.subplots(3, 3, figsize=(30,30))
f = 0
c = 0
for roi in rois:
    m1, m2 = roi[0], roi[1]
    z = np.where((imp_peaks_masses>m1) & (imp_peaks_masses<m2))[0]
    x_axismaldis = np.where((meanmas>m1) & (meanmas<m2))[0]
    x_axismasses = meanmas[x_axismaldis]
    ax[f,c].plot(x_axismasses, np.mean(data['maldis'][p027, :][:, x_axismaldis], axis=0), color='purple', ls=':', label="RT027")
    ax[f,c].plot(x_axismasses, np.mean(data['maldis'][p181, :][:, x_axismaldis], axis=0), color='blue', ls='-', label="RT181")
    ax[f,c].plot(x_axismasses, np.mean(data['maldis'][pxxx, :][:, x_axismaldis], axis=0), color='green', ls='-.', label="RTXXX")
    ax[f,c].axvline(x=imp_peaks_masses[z][0], color='red', alpha=0.2)
    ax[f,c].axvline(x=imp_peaks_masses[z][-1], color='red', alpha=0.2)
    ax[f,c].text(imp_peaks_masses[z][0], 0 ,str(np.round(imp_peaks_masses[z][0],2)), rotation=90)
    ax[f,c].text(imp_peaks_masses[z][-1], 0 ,str(np.round(imp_peaks_masses[z][-1],2)), rotation=90)
    ax[f,c].legend(loc='best')
    f+=1
    if f%3==0: 
        f=0
        c+=1

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines[0:3], labels[0:3])
plt.tight_layout()
plt.show()
    

# Build the grid search ONLY WITH IMPORTANT PEAKS
dfrst_peaks = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid_peaks = GridSearchCV(estimator=dfrst_peaks, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results_peaks = grid_peaks.fit(x_train[:, imp_peaks_loc], y_train)

best_clf = grid_results_peaks.best_estimator_
y_pred_maldi = best_clf.predict(x_test[:, imp_peaks_loc])

acc_maldi_peaks = balanced_accuracy_score(y_test, y_pred_maldi)
print(acc_maldi_peaks)


results = {'imp_peaks_location': imp_peaks_loc, 
                'imp_peaks_masses': imp_peaks_masses,
                'accuracy_fullmaldi': acc_maldi,
                'accuracy_imp_peaks': acc_maldi_peaks,
                'fullRFmodel': grid_results,
                'RFonlyusingpeaks': grid_results_peaks}

with open('./results/RandomForest_noalignment.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)