import json
import pandas as pd
import numpy as np

# Load data from json
with open("data/train.json", "r") as handle:
    test_d = json.load(handle)
    test_df = pd.DataFrame(json.loads(test_d)).T

print(test_df)

# Split a string by ' ', get the two first words and join them
test_df['category'] = test_df['strain'].apply(lambda x: ' '.join(x.split(' ')[0:2]))

# Remove all characters after '_' in the category column
test_df['category'] = test_df['category'].apply(lambda x: x.split('_')[0])

# Categorise to numbers: Staphilococcus aureus-> 0,Acinetobacter baumani >1, Pseudomonas aeruginosa->2, Pseudomonas fluorescence->3, Pseudomonas spp->4
test_df['category_num'] = test_df['category'].apply(lambda x: 0 if x == 'Staphilococcus aureus' else 1 if x == 'Acinetobacter baumani' else 2 if x == 'Pseudomonas aeruginosa' else 3 if x == 'Pseudomonas fluorescence' else 4)

def create_speactr(mz, intens):
    spec = []
    for i in range(200, 1750):
        if i in mz:
            spec.append(intens[mz.index(i)])
        else:
            spec.append(0)
    return spec

def prepocess_data(data):
    data['mz'] = data['m/z'].apply(lambda x: [int(x_i // 10) for x_i in x])
    data['intens'] = data.apply(lambda d: create_speactr(d['mz'], d['Rel. Intens.']),
                           axis = 1)
    return data

data = prepocess_data(test_df)

# Create the X and y data: X -> 'Intens.' and y -> 'category_num'
X = np.vstack(data['intens'].values)
y = data['category_num'].values

# Split train test 0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a random forest and fit it
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)

# Predict the test data
y_pred = rf.predict(X_test)

# Print the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Calculate balanced accuracy
from sklearn.metrics import balanced_accuracy_score
print(balanced_accuracy_score(y_test, y_pred))

# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))




