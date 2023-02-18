# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix ,classification_report
from sklearn.ensemble import RandomForestClassifier

data  = pd.read_csv("src\kaggle_dataset\dataset.csv")
data_severity = pd.read_csv("src\kaggle_dataset\Symptom-severity.csv")

for ind in data_severity.index:
    continue
   # print(data_sevrity['Symptom'][ind],data_sevrity['weight'][ind])

#convert data_severity to dictionnary
data_dict = data_severity.set_index('Symptom').T.to_dict()

def remove_space_between_word(dataset):
    for col in dataset.columns:
        for i in range(len(dataset[col])):
            if (type(dataset[col][i]) == str ):
                dataset[col][i] = dataset[col][i].strip()
                dataset[col][i] = dataset[col][i].replace(" ", "_")
    return data

new_df = remove_space_between_word(data)
new_df.head()

def enc(dataset):
    for ind in data_severity.index:
        dataset = dataset.replace(data_severity["Symptom"][ind] , data_severity["weight"][ind])
    dataset = dataset.fillna(0) # put empty cell to 0
    dataset = dataset.replace("foul_smell_of_urine" , 5)
    dataset = dataset.replace("dischromic__patches" , 6)
    dataset = dataset.replace("spotting__urination" , 6)
    return dataset

ds=enc(data)
ds.dtypes

def encode_data(dataset , data_dict_weigth):
    cols = dataset.columns
    for columnName in cols:
        for i in range(len(dataset[columnName])):
            try:
            #print(data_dict[data2[columnName][i]]["weight"])
                dataset[columnName][i] = data_dict[dataset[columnName][i]]["weight"]
            except:
                pass
    dataset = dataset.fillna(0) # put empty cell to 0
    dataset = dataset.replace("foul_smell_of_urine" , 5)
    dataset = dataset.replace("dischromic__patches" , 6)
    dataset = dataset.replace("spotting__urination" , 6)
    return dataset
df = encode_data(new_df , data_dict)
df.head()

#check if all Symptoms are replace by their weigth
names = []
for col in df.columns:
    if(col != "Disease"):
        for i in range(len(df[col])):
            if (type(df[col][i]) == str ):
                if df[col][i] not in names :
                    names.append(df[col][i])
                    
print(" no replace are :" , names)  

df_data = df.drop('Disease' , axis =1)
label = data["Disease"]

x_train, x_test, y_train, y_test = train_test_split(df_data, label, shuffle=True, train_size = 0.70)
randomFC = RandomForestClassifier()
randomFC.fit(x_train, y_train)
result = randomFC.predict(x_test)
print(randomFC)
print(classification_report(y_true=y_test, y_pred=result))
print('F1-score% =', f1_score(y_test, result, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, result)*100)

qw=pd.DataFrame([[5,4,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],columns=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
       'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
       'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
       'Symptom_16', 'Symptom_17'])

output = randomFC.predict(qw)

import pickle
filename='model.pkl'
pickle.dump(randomFC,open(filename,'wb'))
clff=pickle.load(open('model.pkl','rb'))
result = clff.predict(x_test)
print(classification_report(y_true=y_test, y_pred=result))
print('F1-score% =', f1_score(y_test, result, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, result)*100)
print(output)