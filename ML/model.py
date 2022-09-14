# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Pickle package
import pickle


dataset = pd.read_csv('./data_set/dataset.csv')

print(dataset)

X_test = dataset[['Soil Moisture']].to_numpy()
print(X_test)
y_test = np.where(dataset['Irrigation']=='Irrigate', 0, 1)
print(y_test)

model = svm.SVC(kernel='linear')
model.fit(X_test, y_test)

def irrigate_or_not_irrigate(soil_moisture):
    if(model.predict([[soil_moisture]]))==0:
        print('Irrigate')
    else:
        print('Not irrigate')

irrigate_or_not_irrigate(0)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
