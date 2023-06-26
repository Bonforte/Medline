#File that trains the RandomForest model and plots the performance indeces

import pandas as pd
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import random
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data  = pd.read_csv("kaggle_dataset\dataset.csv")


# Create useful variables with the loaded data
data.columns = data.columns.str.lower()
symp_cols = data[data.columns[1:]].columns

# Prepare data for training

values = []
for col in symp_cols :
    values = values + list(data[symp_cols[0]].values)

data.fillna("notprovided ",inplace=True)

data.dropna(axis=1,inplace=True,how='all')

data["symptoms"] = data['symptom_1'] + "--" + data['symptom_2'] + "--" + data['symptom_3'] + "--" + data['symptom_4'] + "--" + data['symptom_5'] + "--" + \
        data['symptom_6'] + "--" + data['symptom_7'] + "--" + data['symptom_8'] + "--" + data['symptom_9'] + "--" + data['symptom_10'] + "--" + \
        data['symptom_11'] + "--" + data['symptom_12'] + "--" + data['symptom_13'] + "--" + data['symptom_14'] + "--" + data['symptom_15'] + "--" + \
        data['symptom_16'] + "--" + data['symptom_17']

data = data[["disease","symptoms"]]

counter = Counter(values)
results = pd.Series(dict(counter))
results.sort_values(ascending=True)

# Results.index trebuie randomizat pe modele diferite
results_index = list(results.index)
random.shuffle(results_index)
for item in results_index:
    symptom = item.strip()
    data[f"{symptom}"] = data["symptoms"].apply(lambda x : 1 if symptom in x else 0)

y = data["disease"]
x = data.loc[:, ~data.columns.isin(['disease', 'symptoms'])]

# Training data for random forest model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,shuffle=True, random_state=42,stratify=y)
# logreg = LogisticRegression()
# logreg.fit(x_train, y_train)
randomFC = RandomForestClassifier()
randomFC.fit(x_train, y_train)

y_prediction = randomFC.predict(x_test)
result = confusion_matrix(y_test, y_prediction , normalize='pred')
print(result)
# pyplot.plot(hist.history['accuracy'])
# pyplot.plot(hist.history['mean_squared_error'])
# pyplot.plot(hist.history['mean_absolute_error'])
# pyplot.plot(hist.history['rmse'])
# pyplot.plot(hist.history['mean_absolute_percentage_error'])
# pyplot.plot(hist.history['cosine_proximity'])
# pyplot.show()
df_cm = pd.DataFrame(result, range(40), range(40))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}) # font size
print(accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction))

y_pred = list(y_prediction)
y_test = list(y_test)
for i in range(len(y_prediction)):
    if y_prediction[i] != y_test[i]:
        print(f"{y_prediction[i]} -- {y_test[i]}")

plt.show()