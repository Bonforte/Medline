# File where the training for the symptoms to disease prediction is executed

import pandas as pd
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle


def training_and_data_parsing():
    # Disease prediction training:

    # Load csv and json data
    data  = pd.read_csv("kaggle_dataset\dataset.csv")
    data_severity = pd.read_csv("kaggle_dataset\Symptom-severity.csv")
    data_description = pd.read_csv("kaggle_dataset\symptom_Description.csv")
    data_precaution = pd.read_csv("kaggle_dataset\symptom_precaution.csv")

    with open('json\hospital_struct.json') as json_file:
        doctors_json_db = json.load(json_file) 

    # Create useful variables with the loaded data
    data.columns = data.columns.str.lower()
    symp_cols = data[data.columns[1:]].columns

    unique_symptoms = data_severity["Symptom"].unique()
    symptom_to_severity = data_severity.set_index('Symptom').T.to_dict()

    description_to_disease = data_description.set_index('Disease').T.to_dict()
    description_to_disease =  {k.lower(): v for k, v in description_to_disease.items()}

    disease_to_precautions = dict()
    for index, row in data_precaution.iterrows():
        disease_to_precautions[row['Disease']] = [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]

    disease_to_precautions =  {k.lower(): v for k, v in disease_to_precautions.items()}

    disease_to_symptoms = dict()
    for index, row in data.iterrows():
        disease_to_symptoms[row['disease'].lower().strip()] = [row['symptom_1'], row['symptom_2'], row['symptom_3'], row['symptom_4'], row['symptom_5'], row['symptom_6'], row['symptom_7'],
        row['symptom_8'], row['symptom_9'], row['symptom_10'], row['symptom_11'], row['symptom_12'], row['symptom_13'], row['symptom_14'], row['symptom_15'], row['symptom_16'],
        row['symptom_17']]

    disease_names = []
    for key, value in description_to_disease.items():
        disease_names.append(key)
    disease_names = [disease.title() for disease in disease_names]

    # Remove special characters from disease list
    parsed_disease_names = []
    for disease_index in range(len(disease_names)):
        first_replacement_disease = disease_names[disease_index].replace('(', '').replace(')', '')
        parsed_disease_names.append(first_replacement_disease.replace(' ', '_').lower())

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

    for item in results.index:
        symptom = item.strip()
        data[f"{symptom}"] = data["symptoms"].apply(lambda x : 1 if symptom in x else 0)

    data.drop(["symptoms"],inplace=True,axis=1)

    y = data["disease"]
    x = data.drop(['disease'],axis=1)

    # Training data for random forest model
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state=42,stratify=y)
    symptom_list = list(x_test.columns.values)
    # logreg = LogisticRegression()
    # logreg.fit(x_train, y_train)
    randomFC = RandomForestClassifier()
    randomFC.fit(x_train, y_train)

    # filename='rfcmodel.pkl'
    # pickle.dump(randomFC,open(filename,'wb'))
    # clff=pickle.load(open('model.pkl','rb'))
    # result = clff.predict(x_test)
    # print(classification_report(y_true=y_test, y_pred=result))
    # print('F1-score% =', f1_score(y_test, result, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, result)*100)

    return randomFC, symptom_list, symptom_to_severity, disease_to_symptoms, description_to_disease, disease_to_precautions, doctors_json_db, parsed_disease_names, unique_symptoms, disease_names