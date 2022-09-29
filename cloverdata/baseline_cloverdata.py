import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

with open('data/clover_proccesed_fulldataset.pkl', 'rb') as handle:
    fd = pickle.load(handle)

ir_df = fd['ir']
labels = fd['ribotype']
# We remove the signal bigger than 20k because it is empty
maldi_df = fd['maldi'].iloc[:, :20000]

# ======================= Only using MALDITOF =======================

y3 = np.arange(275)
y3[np.where(labels==27)[0]]=27
y3[np.where(labels==181)[0]]=181
y3[np.where((labels!=181) & (labels!=27))[0]]=3

x_train, x_test, y_train, y_test = train_test_split(maldi_df, y3, train_size=0.6, random_state=0)

max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train, y_train)

# Summarize the results in a readable format
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

imp_peaks = np.where(grid.best_estimator_.feature_importances_>0.015)[0]

dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train.values[:, imp_peaks], y_train)

# Summarize the results in a readable format
best_clf = grid_results.best_estimator_
y_pred_maldi_peaks = best_clf.predict(x_test.values[:, imp_peaks])

acc_maldi_peaks = balanced_accuracy_score(y_test, y_pred_maldi_peaks)
print(acc_maldi_peaks)

# Plot differences in mean and std of the 3 samples in test
# Ribotype 027
mean1 = np.mean(x_test.values[np.where(y_test==27)[0], :], axis=0)
std1 = np.std(x_test.values[np.where(y_test==27)[0], :], axis=0)

# Ribotype 181
mean2 = np.mean(x_test.values[np.where(y_test==181)[0], :], axis=0)
std2 = np.std(x_test.values[np.where(y_test==181)[0], :], axis=0)

# Any other ribotype
mean3 = np.mean(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :], axis=0)
std3 = np.std(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :], axis=0)

plt.figure()
plt.plot(np.arange(len(mean1)), mean1, 'b-', label="Ribotype 027")
plt.fill_between(np.arange(len(mean1)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.plot(np.arange(len(mean1)), mean2, 'r-', label="Ribotype 181")
plt.fill_between(np.arange(len(mean1)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.plot(np.arange(len(mean1)), mean3, 'g-', label="Other ribotypes")
plt.fill_between(np.arange(len(mean1)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.legend()
plt.show()


# Plot differences in mean and std of the 3 samples in test USING ONLY THE IMPORTANT PEAKS
# Ribotype 027
mean1 = np.mean(x_test.values[np.where(y_test==27)[0], :][:, imp_peaks], axis=0)
std1 = np.std(x_test.values[np.where(y_test==27)[0], :][:, imp_peaks], axis=0)

# Ribotype 181
mean2 = np.mean(x_test.values[np.where(y_test==181)[0], :][:, imp_peaks], axis=0)
std2 = np.std(x_test.values[np.where(y_test==181)[0], :][:, imp_peaks], axis=0)

# Any other ribotype
mean3 = np.mean(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :][:, imp_peaks], axis=0)
std3 = np.std(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :][:, imp_peaks], axis=0)

plt.figure()
plt.title("Test samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.fill_between(np.arange(len(mean1)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 181")
plt.fill_between(np.arange(len(mean1)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean3, label="Other ribotypes")
plt.fill_between(np.arange(len(mean1)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.xticks(np.arange(len(mean1)), labels=imp_peaks)
plt.legend()
plt.show()

plt.figure()
plt.title("Test samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.errorbar(np.arange(len(mean1)), mean1, yerr=std1, fmt='o')
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 0181")
plt.errorbar(np.arange(len(mean1)), mean2, yerr=std2, fmt='o')
plt.scatter(np.arange(len(mean1)), mean3, label="Othery ribotypes")
plt.errorbar(np.arange(len(mean1)), mean3, yerr=std3, fmt='o')
plt.xticks(np.arange(len(mean1)), labels=imp_peaks)
plt.legend()
plt.show()

# Plot differences in mean and std of the 3 samples in train USING ONLY THE IMPORTANT PEAKS
# Ribotype 027
mean1 = np.mean(x_train.values[np.where(y_train==27)[0], :][:, imp_peaks], axis=0)
std1 = np.std(x_train.values[np.where(y_train==27)[0], :][:, imp_peaks], axis=0)

# Ribotype 181
mean2 = np.mean(x_train.values[np.where(y_train==181)[0], :][:, imp_peaks], axis=0)
std2 = np.std(x_train.values[np.where(y_train==181)[0], :][:, imp_peaks], axis=0)

# Any other ribotype
mean3 = np.mean(x_train.values[np.where((y_train!=181) & (y_train!=27))[0], :][:, imp_peaks], axis=0)
std3 = np.std(x_train.values[np.where((y_train!=181) & (y_train!=27))[0], :][:, imp_peaks], axis=0)

plt.figure()
plt.title("Train samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.fill_between(np.arange(len(mean1)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 181")
plt.fill_between(np.arange(len(mean1)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean3, label="Other ribotypes")
plt.fill_between(np.arange(len(mean1)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.xticks(np.arange(len(mean1)), labels=imp_peaks)
plt.legend()
plt.show()

plt.figure()
plt.title("TRain samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.errorbar(np.arange(len(mean1)), mean1, yerr=std1, fmt='o')
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 0181")
plt.errorbar(np.arange(len(mean1)), mean2, yerr=std2, fmt='o')
plt.scatter(np.arange(len(mean1)), mean3, label="Othery ribotypes")
plt.errorbar(np.arange(len(mean1)), mean3, yerr=std3, fmt='o')
plt.xticks(np.arange(len(mean1)), labels=imp_peaks)
plt.legend()
plt.show()

# Plot differences in mean and std of the 3 samples in test USING ONLY THE IMPORTANT PEAKS PLUS ANA IMPORTANT PEAKS

imp_peaks_ana = imp_peaks.tolist() + [4932, 4953, 3353, 4990, 6707]

# Ribotype 027
mean1 = np.mean(x_test.values[np.where(y_test==27)[0], :][:, imp_peaks_ana], axis=0)
std1 = np.std(x_test.values[np.where(y_test==27)[0], :][:, imp_peaks_ana], axis=0)

# Ribotype 181
mean2 = np.mean(x_test.values[np.where(y_test==181)[0], :][:, imp_peaks_ana], axis=0)
std2 = np.std(x_test.values[np.where(y_test==181)[0], :][:, imp_peaks_ana], axis=0)

# Any other ribotype
mean3 = np.mean(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :][:, imp_peaks_ana], axis=0)
std3 = np.std(x_test.values[np.where((y_test!=181) & (y_test!=27))[0], :][:, imp_peaks_ana], axis=0)

plt.figure(figsize=[15,10])
plt.title("Test samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.fill_between(np.arange(len(mean1)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 181")
plt.fill_between(np.arange(len(mean1)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean3, label="Other ribotypes")
plt.fill_between(np.arange(len(mean1)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.xticks(np.arange(len(mean1)), labels=imp_peaks_ana)
plt.legend()
plt.show()

plt.figure(figsize=[15,10])
plt.title("Test samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.errorbar(np.arange(len(mean1)), mean1, yerr=std1, fmt='o')
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 0181")
plt.errorbar(np.arange(len(mean1)), mean2, yerr=std2, fmt='o')
plt.scatter(np.arange(len(mean1)), mean3, label="Othery ribotypes")
plt.errorbar(np.arange(len(mean1)), mean3, yerr=std3, fmt='o')
plt.xticks(np.arange(len(mean1)), labels=imp_peaks_ana)
plt.legend()
plt.show()

# Plot differences in mean and std of the 3 samples in train USING ONLY THE IMPORTANT PEAKS
# Ribotype 027
mean1 = np.mean(x_train.values[np.where(y_train==27)[0], :][:, imp_peaks_ana], axis=0)
std1 = np.std(x_train.values[np.where(y_train==27)[0], :][:, imp_peaks_ana], axis=0)

# Ribotype 181
mean2 = np.mean(x_train.values[np.where(y_train==181)[0], :][:, imp_peaks_ana], axis=0)
std2 = np.std(x_train.values[np.where(y_train==181)[0], :][:, imp_peaks_ana], axis=0)

# Any other ribotype
mean3 = np.mean(x_train.values[np.where((y_train!=181) & (y_train!=27))[0], :][:, imp_peaks_ana], axis=0)
std3 = np.std(x_train.values[np.where((y_train!=181) & (y_train!=27))[0], :][:, imp_peaks_ana], axis=0)

plt.figure(figsize=[15,10])
plt.title("Train samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.fill_between(np.arange(len(mean1)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 181")
plt.fill_between(np.arange(len(mean1)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.scatter(np.arange(len(mean1)), mean3, label="Other ribotypes")
plt.fill_between(np.arange(len(mean1)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.xticks(np.arange(len(mean1)), labels=imp_peaks_ana)
plt.legend()
plt.show()

plt.figure(figsize=[15,10])
plt.title("TRain samples")
plt.scatter(np.arange(len(mean1)), mean1, label="Ribotype 027")
plt.errorbar(np.arange(len(mean1)), mean1, yerr=std1, fmt='o')
plt.scatter(np.arange(len(mean1)), mean2, label="Ribotype 0181")
plt.errorbar(np.arange(len(mean1)), mean2, yerr=std2, fmt='o')
plt.scatter(np.arange(len(mean1)), mean3, label="Othery ribotypes")
plt.errorbar(np.arange(len(mean1)), mean3, yerr=std3, fmt='o')
plt.xticks(np.arange(len(mean1)), labels=imp_peaks_ana)
plt.legend()
plt.show()


# ======================= Only using IR =======================
x_train, x_test, y_train, y_test = train_test_split(ir_df, y3, train_size=0.6, random_state=0)

max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train, y_train)

# Summarize the results in a readable format
best_clf = grid_results.best_estimator_
y_pred_ir = best_clf.predict(x_test)

acc_ir = balanced_accuracy_score(y_test, y_pred_ir)
print(acc_ir)

plt.figure()
plt.title("Using only IR")
plt.plot(np.mean(x_test, axis=0), label='Signal mean in test')
plt.plot(grid.best_estimator_.feature_importances_, label='Feature importances', alpha=0.2)
plt.legend()
plt.show()

# ======================= Combining both spectras directly naively =======================

maldir = np.hstack((maldi_df.values, ir_df.values*1e-2))

x_train, x_test, y_train, y_test = train_test_split(maldir, y3, train_size=0.6, random_state=0)

max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train, y_train)

# Summarize the results in a readable format
best_clf = grid_results.best_estimator_
y_pred_maldir = best_clf.predict(x_test)

acc_maldir = balanced_accuracy_score(y_test, y_pred_maldir)
print(acc_maldir)

plt.figure()
plt.title("Concatenating MALDI and IR")
plt.plot(np.mean(x_test, axis=0), label='Signal mean in test')
plt.plot(grid.best_estimator_.feature_importances_, label='Feature importances', alpha=0.2)
plt.legend()
plt.show()

# ======================= Combining both spectras only important peaks of maldi =======================

maldir = np.hstack((maldi_df.values[:, imp_peaks], ir_df.values*1e-2))

x_train, x_test, y_train, y_test = train_test_split(maldir, y3, train_size=0.6, random_state=0)

max_depth=[2, 8, 16]
n_estimators = [64, 128, 256]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

# Build the grid search
dfrst = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
grid = GridSearchCV(estimator=dfrst, param_grid=param_grid, scoring='balanced_accuracy', cv = 5, verbose=2)
grid_results = grid.fit(x_train, y_train)

# Summarize the results in a readable format
best_clf = grid_results.best_estimator_
y_pred_maldir_peaks = best_clf.predict(x_test)

acc_maldir_peaks = balanced_accuracy_score(y_test, y_pred_maldir_peaks)
print(acc_maldir_peaks)

plt.figure()
plt.title("Concatenating MALDI 8 peaks and IR")
plt.plot(np.mean(x_test, axis=0), label='Signal mean in test')
plt.plot(grid.best_estimator_.feature_importances_, label='Feature importances', alpha=0.2)
plt.legend()
plt.show()

# ======================= Final results ======================
plt.figure()
plt.bar(['MALDI', 'MALDI 8 peaks', 'IR', 'MALDI + IR', 'MALDI 8 peaks + IR'], [acc_maldi, acc_maldi_peaks, acc_ir, acc_maldir, acc_maldir_peaks])
plt.show()