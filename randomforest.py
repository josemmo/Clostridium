from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
import numpy as np
import pickle

with open('data/MALDI/data_processedMALDIQuant.pickle', 'rb') as handle:
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

imp_peaks_loc = np.where(grid.best_estimator_.feature_importances_>10*np.std(grid.best_estimator_.feature_importances_))[0]
imp_peaks_masses = np.mean(np.vstack(masses), axis=0)[imp_peaks_loc]

# Build the grid search ONLY WITH IMPORTANT PEAKS
dfrst_peaks = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid_peaks = GridSearchCV(estimator=dfrst_peaks, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results_peaks = grid_peaks.fit(x_train[:, imp_peaks_loc], y_train)

best_clf = grid_results_peaks.best_estimator_
y_pred_maldi = best_clf.predict(x_test[:, imp_peaks_loc])

acc_maldi_peaks = balanced_accuracy_score(y_test, y_pred_maldi)
print(acc_maldi_peaks)

key = 'run'+str(c)
results[key] = {'imp_peaks_location': imp_peaks_loc, 
                'imp_peaks_masses': imp_peaks_masses,
                'accuracy_fullmaldi': acc_maldi,
                'accuracy_imp_peaks': acc_maldi_peaks}

imp_peaks_location = []
imp_peaks_masses = []
accuracy_fullmaldi = []
accuracy_imp_peaks = []
for c in range(9):
    key = 'run'+str(c)
    imp_peaks_masses.append(results[key]['imp_peaks_masses'])
    imp_peaks_location.append(results[key]['imp_peaks_location'])
    accuracy_fullmaldi.append(results[key]['accuracy_fullmaldi'])
    accuracy_imp_peaks.append(results[key]['accuracy_imp_peaks'])

interesant_peaks_masses = {'roi1': [2400,2500],
                    'roi2': [3300,3400],
                    'roi3': [4650,4750],
                    'roi4': [4900,5000],
                    'roi5': [4950,5050],
                    'roi6': [5000,5100],
                    'roi7': [6600,6700],
                    'roi8': [6700,6800],
                    'roi9': [12600,12700],
                    'roi10': [14800,14900]}

interesant_peaks_masses = {'roi1': [1040,1150],
                    'roi2': [2750,2850],
                    'roi3': [5100,5200],
                    'roi4': [5400,5500],
                    'roi5': [5500,5600],
                    'roi6': [5600,5700],
                    'roi7': [7800,7900],
                    'roi8': [7900,8000],
                    'roi9': [14400,14500],
                    'roi10': [16500,16600]}

# CHECK THE PEAKS IN TEST
for i in range(1,11):
    randominit = 0
    key1 = 'roi'+str(i)
    key2 = 'run1'
    roi0 = interesant_peaks_masses[key1][0]
    roi1 = interesant_peaks_masses[key1][1]
    imppeaksloc_currentroi = imp_peaks_location[randominit][np.where((imp_peaks_location[randominit]>roi0) & (imp_peaks_location[randominit]<roi1))[0]]
    imppeakmasses_currentroi = imp_peaks_masses[randominit][np.where((imp_peaks_location[randominit]>roi0) & (imp_peaks_location[randominit]<roi1))[0]]
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title("MALDI important peaks to classify between RT in TEST")
    ax.scatter(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==0)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for 027')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==0)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_test[np.where(y_test==0)[0], :][:, imppeaksloc_currentroi], axis=0))
    
    ax.scatter(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==1)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for 181')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==1)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_test[np.where(y_test==1)[0], :][:, imppeaksloc_currentroi], axis=0))
    
    ax.scatter(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==2)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for others')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_test[np.where(y_test==2)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_test[np.where(y_test==2)[0], :][:, imppeaksloc_currentroi], axis=0))

    ax.set_xticks(imppeaksloc_currentroi)
    ax.set_xticklabels(np.round(imppeakmasses_currentroi,2), rotation=45)
    ax.legend()

# CHECK THE PEAKS IN TRAIN
for i in range(1,11):
    randominit = 0
    key1 = 'roi'+str(i)
    key2 = 'run1'
    roi0 = interesant_peaks_masses[key1][0]
    roi1 = interesant_peaks_masses[key1][1]
    imppeaksloc_currentroi = imp_peaks_location[randominit][np.where((imp_peaks_location[randominit]>roi0) & (imp_peaks_location[randominit]<roi1))[0]]
    imppeakmasses_currentroi = imp_peaks_masses[randominit][np.where((imp_peaks_location[randominit]>roi0) & (imp_peaks_location[randominit]<roi1))[0]]
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title("MALDI important peaks to classify between RT in TEST")
    ax.scatter(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==0)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for 027')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==0)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_train[np.where(y_train==0)[0], :][:, imppeaksloc_currentroi], axis=0))
    
    ax.scatter(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==1)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for 181')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==1)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_train[np.where(y_train==1)[0], :][:, imppeaksloc_currentroi], axis=0))
    
    ax.scatter(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==2)[0], :][:, imppeaksloc_currentroi], axis=0), label='Peak mean for others')
    ax.errorbar(imppeaksloc_currentroi, np.mean(x_train[np.where(y_train==2)[0], :][:, imppeaksloc_currentroi], axis=0), yerr=np.std(x_train[np.where(y_train==2)[0], :][:, imppeaksloc_currentroi], axis=0))

    ax.set_xticks(imppeaksloc_currentroi)
    ax.set_xticklabels(np.round(imppeakmasses_currentroi,2), rotation=45)
    ax.legend()


roi0=2480
roi1=2500
plt.plot(masses[0][np.where((masses[0]>roi0) & (masses[0]<roi1))[0]],np.mean(maldis[:, np.where((masses[0]>roi0) & (masses[0]<roi1))[0]][np.where(y3==0)[0]], axis=0), label="RT027")
plt.plot(masses[0][np.where((masses[0]>roi0) & (masses[0]<roi1))[0]],np.mean(maldis[:, np.where((masses[0]>roi0) & (masses[0]<roi1))[0]][np.where(y3==1)[0]], axis=0), label="RT181")
plt.plot(masses[0][np.where((masses[0]>roi0) & (masses[0]<roi1))[0]],np.mean(maldis[:, np.where((masses[0]>roi0) & (masses[0]<roi1))[0]][np.where(y3==2)[0]], axis=0), label="RTXXX")
plt.legend()
plt.show()



plt.figure(figsize=(20,10))
plt.title("Using MALDI important peaks")
plt.plot(np.arange(len(imp_peaks_loc)),np.mean(x_test[np.where(y_test==0)[0], :][:, imp_peaks_loc], axis=0), label='Peak mean for 027')
plt.plot(np.arange(len(imp_peaks_loc)),np.mean(x_test[np.where(y_test==1)[0], :][:, imp_peaks_loc], axis=0), label='Peak mean for 181')
plt.plot(np.arange(len(imp_peaks_loc)),np.mean(x_test[np.where(y_test==2)[0], :][:, imp_peaks_loc], axis=0), label='Peak mean for others')
plt.xticks(np.arange(len(imp_peaks_loc)), labels=np.round(imp_peaks_masses,2), rotation=45)
plt.legend()
plt.grid()
plt.show()

newx = np.arange(0, imp_peaks_loc[-2]+1)
new27 = np.zeros(newx.shape)
new181 = np.zeros(newx.shape)
newothers = np.zeros(newx.shape)

new27[imp_peaks_loc[:-1]] = np.mean(x_test[np.where(y_test==0)[0], :][:, imp_peaks_loc[:-1]], axis=0)
new181[imp_peaks_loc[:-1]] = np.mean(x_test[np.where(y_test==1)[0], :][:, imp_peaks_loc[:-1]], axis=0)
newothers[imp_peaks_loc[:-1]] = np.mean(x_test[np.where(y_test==2)[0], :][:, imp_peaks_loc[:-1]], axis=0)

fig, ax = plt.subplots(figsize=(20,10))
ax.set_title("Using MALDI important peaks")
ax.plot(np.arange(len(new27)),new27, label='Peak mean for 027')
ax.plot(np.arange(len(new181)),new181, label='Peak mean for 181')
ax.plot(np.arange(len(newothers)),newothers, label='Peak mean for others')

ax.set_xticks(imp_peaks_loc)
ax.set_xticklabels(np.round(imp_peaks_masses,2))

plt.legend()
plt.grid()
plt.show()

# Split previous plot in zoom 1
fig, ax = plt.subplots(figsize=(20,10))
ax.set_title("Using MALDI important peaks")
ax.plot(np.arange(len(new27))[2600:3000],new27[2600:3000], label='Peak mean for 027')
ax.plot(np.arange(len(new181))[2600:3000]+5,new181[2600:3000], label='Peak mean for 181')
ax.plot(np.arange(len(newothers))[2600:3000]+10,newothers[2600:3000], label='Peak mean for others')

ax.set_xticks(imp_peaks_loc[0:12])
ax.set_xticklabels(np.round(imp_peaks_masses,2)[0:12], rotation=45)

plt.legend()
plt.grid()
plt.show()

# Split previous plot in zoom 2
fig, ax = plt.subplots(figsize=(20,10))
ax.set_title("Using MALDI important peaks")
ax.plot(np.arange(len(new27))[4500:5700],new27[4500:5700], label='Peak mean for 027')
ax.plot(np.arange(len(new181))[4500:5700]+5,new181[4500:5700], label='Peak mean for 181')
ax.plot(np.arange(len(newothers))[4500:5700]+10,newothers[4500:5700], label='Peak mean for others')

ax.set_xticks(imp_peaks_loc[13:19])
ax.set_xticklabels(np.round(imp_peaks_masses,2)[13:19], rotation=45)

plt.legend()
plt.grid()
plt.show()

# Split previous plot in zoom 3
fig, ax = plt.subplots(figsize=(20,10))
ax.set_title("Using MALDI important peaks")
ax.plot(np.arange(len(new27))[7700:8000],new27[7700:8000], label='Peak mean for 027')
ax.plot(np.arange(len(new181))[7700:8000]+5,new181[7700:8000], label='Peak mean for 181')
ax.plot(np.arange(len(newothers))[7700:8000]+10,newothers[7700:8000], label='Peak mean for others')

ax.set_xticks(imp_peaks_loc[21:])
ax.set_xticklabels(np.round(imp_peaks_masses,2)[21:], rotation=45)

plt.legend()
plt.grid()
plt.show()







