import numpy as np
import pickle
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt

with open('data/MALDI/data_processedMALDIQuant_noalign.pickle', 'rb') as handle:
    data = pickle.load(handle)

masses = data['masses']
meanmas = np.mean(masses, axis=0)

with open('./results/using_only_peaksRF.pickle', 'rb') as handle:
    results_peaks = pickle.load(handle)

with open('./results/sshiba_reg_noalignment.pickle', 'rb') as handle:
    results_full = pickle.load(handle)
#===================FULL MALDI===================

y_train = results_full['y_train']
y_test = results_full['y_test']
model = results_full['model']

Wmaldi = model.q_dist.W[0]
Wrt = model.q_dist.W[1]

y_pred_ohe=model.t[1]['mean'][y_train.shape[0]:,:]
y_pred = np.argmax(y_pred_ohe, axis=1)

acc1 = balanced_accuracy_score(y_test, y_pred)

plt.plot(np.mean(np.abs(Wmaldi['mean']), axis=1))
plt.show()


#==================test a sample====================
from sklearn.metrics import average_precision_score

print("Test all samples")
maldis = data['maldis']*1e5
maldis_sshiba = model.struct_data(X=maldis, method="reg")
pred = model.predict([0,1], [1], maldis_sshiba)
acc_total=balanced_accuracy_score(data['labels_3cat'], np.argmax(pred['output_view1']['mean_x'], axis=1))
aps = average_precision_score(data, np.max(pred['output_view1']['mean_x'], axis=1))


print("test only train samples")
maldis = results_full['x_train']
maldis_sshiba = model.struct_data(X=maldis, method="reg")
pred = model.predict([0,1], [1], maldis_sshiba)
acc_train=balanced_accuracy_score(y_train, np.argmax(pred['output_view1']['mean_x'], axis=1))


print("Test test samples")
maldis = results_full['x_test']
maldis_sshiba = model.struct_data(X=maldis, method="reg")
pred = model.predict([0,1], [1], maldis_sshiba)
acc_test=balanced_accuracy_score(y_test, np.argmax(pred['output_view1']['mean_x'], axis=1))


#==================Show fancy results====================

peaks_weights = np.mean(np.abs(Wmaldi['mean']), axis=1)

imp_peaks_loc = np.where(peaks_weights>1.5)[0]
imp_peaks_masses = meanmas[imp_peaks_loc]

p027 = np.where(data['labels_3cat']==0)[0]
p181 = np.where(data['labels_3cat']==1)[0]
pxxx = np.where(data['labels_3cat']==2)[0]
rois = [[2600,2650], [3250,3300], [3600,3650], [3950,4000],  [4900,5000],  [7250,7280], [7900, 7950]]
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



#===================ONLY PEAKS===================



