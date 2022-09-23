import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('data/clover_proccesed_fulldataset.pkl', 'rb') as handle:
    fd = pickle.load(handle)

ir_df = fd['ir']
labels = fd['ribotype']
maldi_df = fd['maldi']
# ======================= Analyze IR data =======================
# All data
mean = np.mean(ir_df.values, axis=0)
std = np.std(ir_df.values, axis=0)

plt.plot(np.arange(len(mean)), mean, 'b-', label="All IR mean")
plt.fill_between(np.arange(len(mean)), mean - std, mean+std, color='b', alpha=0.1)
plt.legend()
plt.show()

# Ribotype 027
mean1 = np.mean(ir_df.values[np.where(labels==27)[0], :], axis=0)
std1 = np.std(ir_df.values[np.where(labels==27)[0], :], axis=0)

# Ribotype 181
mean2 = np.mean(ir_df.values[np.where(labels==181)[0], :], axis=0)
std2 = np.std(ir_df.values[np.where(labels==181)[0], :], axis=0)

# Any other ribotype
mean3 = np.mean(ir_df.values[np.where((labels!=181) & (labels!=27))[0], :], axis=0)
std3 = np.std(ir_df.values[np.where((labels!=181) & (labels!=27))[0], :], axis=0)


plt.figure()
plt.plot(np.arange(len(mean)), mean1, 'b-', label="Ribotype 027")
plt.fill_between(np.arange(len(mean)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.plot(np.arange(len(mean)), mean2, 'r-', label="Ribotype 181")
plt.fill_between(np.arange(len(mean)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.plot(np.arange(len(mean)), mean3, 'g-', label="Other ribotypes")
plt.fill_between(np.arange(len(mean)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.legend()
plt.show()

# ======================= Analyze MALDI data =======================
mean = np.mean(maldi_df.values, axis=0)
std = np.std(maldi_df.values, axis=0)

plt.plot(np.arange(len(mean)), mean, 'b-', label="All MALDI mean")
plt.fill_between(np.arange(len(mean)), mean - std, mean+std, color='b', alpha=0.1)
plt.legend()
plt.show()

# We remove all signal plus 20k because it is empty
maldi_df = maldi_df.iloc[:, :20000]
mean = np.mean(maldi_df.values, axis=0)
std = np.std(maldi_df.values, axis=0)

plt.plot(np.arange(len(mean)), mean, 'b-', label="All MALDI mean")
plt.fill_between(np.arange(len(mean)), mean - std, mean+std, color='b', alpha=0.1)
plt.legend()
plt.show()

# Ribotype 027
mean1 = np.mean(maldi_df.values[np.where(labels==27)[0], :], axis=0)
std1 = np.std(maldi_df.values[np.where(labels==27)[0], :], axis=0)

# Ribotype 181
mean2 = np.mean(maldi_df.values[np.where(labels==181)[0], :], axis=0)
std2 = np.std(maldi_df.values[np.where(labels==181)[0], :], axis=0)

# Any other ribotype
mean3 = np.mean(maldi_df.values[np.where((labels!=181) & (labels!=27))[0], :], axis=0)
std3 = np.std(maldi_df.values[np.where((labels!=181) & (labels!=27))[0], :], axis=0)


plt.figure()
plt.plot(np.arange(len(mean)), mean1, 'b-', label="Ribotype 027")
plt.fill_between(np.arange(len(mean)), mean1 - std1, mean1+std1, color='b', alpha=0.2)
plt.plot(np.arange(len(mean)), mean2, 'r-', label="Ribotype 181")
plt.fill_between(np.arange(len(mean)), mean2 - std2, mean2+std2, color='r', alpha=0.2)
plt.plot(np.arange(len(mean)), mean3, 'g-', label="Other ribotypes")
plt.fill_between(np.arange(len(mean)), mean3 - std3, mean3+std3, color='g', alpha=0.2)
plt.legend()
plt.show()