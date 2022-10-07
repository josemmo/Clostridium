import numpy as np
import pickle
import pandas as pd
import os
from datetime import datetime

print("=============================================================")
print("Creating folders to store information...")
try:
    os.makedirs("./results/images")
except FileExistsError:
    pass

print("Images will be stored at ./results/images")
print("Predictions will be stored as an Excel file at ./results/")
print("MALDI-TOFs processed by MALDIquant will be stored at ./results/data_maldiquant")
print("=============================================================")
print("Loading MALDI data...")

maldi_path = "./results/data_maldiquant"
# READ DATA MZML
listOfFiles = list()
for (dirpath, dirnames, filenames_rep) in os.walk(maldi_path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames_rep]

masses = []
maldis = []
ids = []
# CONVERT TO A PANDAS DATAFRAME
for filepath in listOfFiles:
    file = filepath.split("\\")[-1]
    if file == ".DS_Store":
        continue
    m = pd.read_csv(filepath)
    maldis.append(m["intensity"].values[0:18000])
    masses.append(m["mass"].values[0:18000])
    ids.append(file.split("_")[-2])
maldis = np.vstack(maldis)
meanmass = np.mean(np.vstack(masses), axis=0)
print("MALDI data loaded.")
print("Total nº of samples: ", str(len(maldis)))
print("=============================================================")
print("Loading RF model...")

with open('./models/RandomForest_noalignment.pickle', 'rb') as handle:
    rfload = pickle.load(handle)
rf = rfload['fullRFmodel']
rf_peaks = rfload['RFonlyusingpeaks']

print("Predicting using RF model...")

pred_rf = rf.best_estimator_.predict(maldis)
pred_proba_rf = np.round(np.max(rf.best_estimator_.predict_proba(maldis), axis=1), 2)-0.01

pred_rfpeaks = rf_peaks.best_estimator_.predict(maldis[:, rfload['imp_peaks_location']])
pred_proba_rfpeaks = np.round(np.max(rf_peaks.best_estimator_.predict_proba(maldis[:,rfload['imp_peaks_location']]), axis=1), 2)-0.01
print("=============================================================")
print("Loading SSHIBA model...")
print("This will take longer, wait a little bit.")
with open('./models/sshiba_reg_noalignment.pickle', 'rb') as handle:
    ss = pickle.load(handle)
print("SSHIBA loaded!")
print("Predicting using SSHIBA...")
model = ss['model']
maldis_sshiba = model.struct_data(X=maldis, method="reg")
pred = model.predict([0,1], [1], maldis_sshiba)

pred_ss = np.argmax(pred['output_view1']['mean_x'], axis=1)
pred_proba_ss = np.round(np.max(pred['output_view1']['mean_x'], axis=1), 2)
print("=============================================================")

print("Creating excel with results..")
results = {"ID": ids, "Pred RF full": pred_rf, "Prob RF full": pred_proba_rf, "Pred RF peaks": pred_rfpeaks, "Prob RF peaks": pred_proba_rfpeaks,
           "Pred SSHIBA": pred_ss, "Prob SSHIBA": pred_proba_ss, "Real RT": np.nan*np.zeros(pred_proba_rf.shape)}
df = pd.DataFrame(results)
df = df.set_index("ID")
df = df.replace(2, "Other").replace(0, "RT027").replace(1, "RT181")

t= datetime.timestamp(datetime.now())

name_df = "./results/prediction_"+str(t)+".xlsx"
df.to_excel(name_df)

print("Excel created and stored at: "+name_df)
print("=============================================================")


print ("Plotting ROIs for each MALDI-TOF...")
from matplotlib import pyplot as plt
z1 = np.where((meanmass > 2600) & (meanmass < 2700)) 
z2 = np.where((meanmass > 3260) & (meanmass < 3360)) 
z3 = np.where((meanmass > 4250) & (meanmass < 4350)) 
z4 = np.where((meanmass > 4900) & (meanmass < 5000)) 
z5 = np.where((meanmass > 6600) & (meanmass < 6700)) 
z6 = np.where((meanmass > 6700) & (meanmass < 6800)) 

zs = [z1, z2, z3, z4, z5, z6]

for i, m in enumerate(maldis):
    fig, ax = plt.subplots(2, 3, figsize=(30,10))
    f, c = 0,0
    for j,z in enumerate(zs):
        ax[f, c].plot(meanmass[z], m[z])
        ax[f,c].set_title("ROI "+str(j)+": from "+str(np.round(meanmass[z][0],2))+"Da to "+str(np.round(meanmass[z][-1],2))+"Da")
        c+=1
        if c==3:
            c=0
            f+=1
    fig.suptitle('ROIs of '+str(ids[i])+"'s MALDI-TOF")
    pathsavepng = './results/images/rois_'+str(ids[i])+'.png'
    fig.savefig(pathsavepng, format='png', dpi=300)

print("Images of ROIs stored at "+pathsavepng)