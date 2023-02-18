import random
import json
import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix ,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.models import load_model

import spacy
import pytextrank
from collections import Counter
from string import punctuation
import difflib
import json


# Disease prediction training:
data  = pd.read_csv("src\kaggle_dataset\dataset.csv")
data_severity = pd.read_csv("src\kaggle_dataset\Symptom-severity.csv")
data_description = pd.read_csv("src\kaggle_dataset\symptom_Description.csv")
data_precaution = pd.read_csv("src\kaggle_dataset\symptom_precaution.csv")

with open('src\json\hospital_struct.json') as json_file:
    doctors_json_db = json.load(json_file) 


unique_symptoms = data_severity["Symptom"].unique()

data_dict = data_severity.set_index('Symptom').T.to_dict()

weight_to_symptom = data_dict

description_to_disease = data_description.set_index('Disease').T.to_dict()

disease_names = []

for key, value in description_to_disease.items():
    disease_names.append(key)

disease_names = [disease.title() for disease in disease_names]

parsed_disease_names = []
for disease_index in range(len(disease_names)):
    first_replacement_disease = disease_names[disease_index].replace('(', '').replace(')', '')
    parsed_disease_names.append(first_replacement_disease.replace(' ', '_').lower())

description_to_disease =  {k.lower(): v for k, v in description_to_disease.items()}

disease_to_precautions = dict()

for index, row in data_precaution.iterrows():
    disease_to_precautions[row['Disease']] = [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]

disease_to_precautions =  {k.lower(): v for k, v in disease_to_precautions.items()}


data.columns = data.columns.str.lower()

symp_cols = data[data.columns[1:]].columns

values = []
for col in symp_cols :
    values = values + list(data[symp_cols[0]].values)

data.fillna(" notprovided ",inplace=True)

symp_cols = data[data.columns[1:]].columns

data.dropna(axis=1,inplace=True,how='all')

data["symptoms"] = data['symptom_1'] + "--" + data['symptom_2'] + "--" + data['symptom_3'] + "--" + data['symptom_4'] + "--" + data['symptom_5'] + "--" + \
        data['symptom_6'] + "--" + data['symptom_7'] + "--" + data['symptom_8'] + "--" + data['symptom_9'] + "--" + data['symptom_10'] + "--" + \
        data['symptom_11'] + "--" + data['symptom_12'] + "--" + data['symptom_13'] + "--" + data['symptom_14'] + "--" + data['symptom_15'] + "--" + \
        data['symptom_16'] + "--" + data['symptom_17']

data = data[["disease","symptoms"]]

data["symptoms"].apply(lambda x: 1 if "itching" in x else 0)

counter = Counter(values)
results = pd.Series(dict(counter))
results.sort_values(ascending=True).plot(kind='barh',figsize=(20,8))

for item in results.index:
    symptom = item.strip()
    data[f"{symptom}"] = data["symptoms"].apply(lambda x : 1 if symptom in x else 0)

data.drop(["symptoms"],inplace=True,axis=1)

y = data["disease"]
x = data.drop(['disease'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,shuffle=True, random_state=42,stratify=y)
symptom_list = list(x_test.columns.values)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
randomFC = RandomForestClassifier()
randomFC.fit(x_train, y_train)


#Create chatbot logic:

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('src\json\intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    probability = intents_list[0]['probability']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result, probability

def get_description(disease):
    print('I will provide a description for ' + str(disease) + '.')
    print(description_to_disease[disease.lower()]['Description'])

def get_precautions(disease):
    print('I will provide precautions for ' + str(disease) + '.')

    for precaution in disease_to_precautions[disease.lower()]:  
        print('- ' + precaution + ';')

def get_doctors(disease):
    print('I will provide proffessional help regarding ' + str(disease) + '.')
    
    for hospital, value in doctors_json_db.items():
        for department, doctors in value.get('departments').items():
            if disease.lower() in [x.lower() for x in doctors.get('diseases')]:
                print('Help can be found at ' + hospital + '.')
                print('It has various departments, but the one of your interest will be: ' + department + '.')
                print('The names of the doctors that can help you here are ' + ', '.join(doctors.get('doctors')) + '.')

def get_interest_words(message):
    interest_words = []
    nlp = spacy.load("en_core_web_sm")

    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
        doc = nlp(text.lower()) 
        for token in doc:
            if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                continue
            if(token.pos_ in pos_tag):
                result.append(token.text)
        return result

    output = set(get_hotwords(message))

    most_common_list = Counter(output).most_common(10)

    for item in most_common_list:
        interest_words.append(item[0])

    return interest_words

def symptom_prediction_route(message, unique_symptoms, settled_symptoms, weight_to_symptom):

    interest_words = get_interest_words(message)

    detected_symptoms = []
    settled_symptoms = []

    concatenated_message = message.replace(' ', '_')
    for symptom in unique_symptoms:
            if symptom in concatenated_message:
                settled_symptoms.append(symptom)
    for word in interest_words:
        found_symptoms = difflib.get_close_matches(word, unique_symptoms)  
        
        if word in unique_symptoms:
            settled_symptoms.append(word)
            continue

        for symptom in unique_symptoms:
            if word in symptom:
                detected_symptoms.append(symptom)
        if found_symptoms:
            detected_symptoms.extend(found_symptoms)

    detected_symptoms = list( dict.fromkeys(detected_symptoms) )
    settled_symptoms = list( dict.fromkeys(settled_symptoms) )

    print(f'I have understood the following symptoms: {settled_symptoms} and recorded them') if settled_symptoms else print('I have not recorded any symptoms.')

    if detected_symptoms:
        print('There might be some symptoms I have not understood.')

        detected_symptoms_string = ', '.join(detected_symptoms)
        print(f'If you reffered to these ones:{detected_symptoms_string}')
        print('Please choose the ones that are of your interest just like you did before. If not, send a negative message.')

        wanted_symptoms_input = input('')
        wanted_symptoms_answer_intent = predict_class(wanted_symptoms_input)
        if wanted_symptoms_answer_intent[0]['intent'] == 'symptom':  
            interest_words = get_interest_words(wanted_symptoms_input)

            concatenated_message = wanted_symptoms_input.replace(' ', '_')
            for symptom in unique_symptoms:
                    if symptom in concatenated_message:
                        settled_symptoms.append(symptom)

            for word in interest_words:
                if word in unique_symptoms:
                    settled_symptoms.append(word)
        elif wanted_symptoms_answer_intent[0]['intent'] != 'negative':
            print('I will start from the beginning.')
            return
        settled_symptoms = list( dict.fromkeys(settled_symptoms) )


    #Test:
    print(settled_symptoms)
    ##

    print('Do you have any other symptoms?')
    while True:
        other_symptoms_input = input('')
        other_symptoms_answer_intent = predict_class(other_symptoms_input)
        print(other_symptoms_answer_intent[0]['intent'])

        if other_symptoms_answer_intent[0]['intent'] == 'affirmative':
            print('Please state them.')
            other_symptoms_message = input('')

            interest_words = get_interest_words(other_symptoms_message)

            detected_symptoms = []

            concatenated_message = other_symptoms_message.replace(' ', '_')
            for symptom in unique_symptoms:
                    if symptom in concatenated_message:
                        settled_symptoms.append(symptom)
            for word in interest_words:
                found_symptoms = difflib.get_close_matches(word, unique_symptoms)  
                
                if word in unique_symptoms:
                    settled_symptoms.append(word)
                    continue

                for symptom in unique_symptoms:
                    if word in symptom:
                        detected_symptoms.append(symptom)

                if found_symptoms:
                    detected_symptoms.extend(found_symptoms)

            detected_symptoms = list( dict.fromkeys(detected_symptoms) )
            settled_symptoms = list( dict.fromkeys(settled_symptoms) )

            print(f'I have understood the following symptoms: {settled_symptoms} and recorded them') if settled_symptoms else print('I have not recorded any symptoms.')

            if detected_symptoms:
                print('There might be some symptoms I have not understood.')

                detected_symptoms_string = ', '.join(detected_symptoms)
                print(f'If you reffered to these ones:{detected_symptoms_string}')
                print('Please choose the ones that are of your interest with words separated by commas. If not, just send me a negative message.')

                wanted_symptoms_input = input('')
                wanted_symptoms_answer_intent = predict_class(wanted_symptoms_input)
                if wanted_symptoms_answer_intent[0]['intent'] == 'symptom':  
                    interest_words = get_interest_words(wanted_symptoms_input)

                    concatenated_message = wanted_symptoms_input.replace(' ', '_')
                    for symptom in unique_symptoms:
                            if symptom in concatenated_message:
                                settled_symptoms.append(symptom)

                    for word in interest_words:
                        if word in unique_symptoms:
                            settled_symptoms.append(word)
                elif wanted_symptoms_answer_intent[0]['intent'] != 'negative':
                    print('I will start from the beginning.')
                    return

                settled_symptoms = list( dict.fromkeys(settled_symptoms) )

            #Test:
            print(settled_symptoms)
            ##
            break
        elif other_symptoms_answer_intent[0]['intent'] == 'negative':
            print('I will predict your disease based on your recorded symptoms.')
            break
        elif other_symptoms_answer_intent[0]['intent'] == 'reset':
            print('I will start from the beginning.')
            return
        else:
            print('I have detected neither a yes or a no. Please rephrase.')

    symptom_presence = []
    for symptom in symptom_list:
        if symptom in settled_symptoms:
            symptom_presence.append(1)
        else:
            symptom_presence.append(0)

    qw=pd.DataFrame([symptom_presence],columns=symptom_list)
    no_of_symptoms = len(settled_symptoms)

    output = randomFC.predict(qw)
    output_disease = output[0]

    if not no_of_symptoms:
        print('I cannot predict a disease with no symptoms.')
        return
    else:    
        return_answer = 'The detected disease is: ' + output_disease
        print(return_answer) 

    if no_of_symptoms < 3:
        print('I am not very sure about this prediction as the number of symptoms given is low.')

    print('Do you want to know more about this disease?')
    while True:
        answer_more = input('')
        answer_more_intent = predict_class(answer_more)
    
        if answer_more_intent[0]['intent'] == 'affirmative':
            print('I can provide description, doctors, hospitals or precautions. Which one do you choose?')

            while True:
                answer_desc_or_prec = input('')
                answer_desc_or_prec_intent = predict_class(answer_desc_or_prec)
            
                if answer_desc_or_prec_intent[0]['intent'] == 'disease_info':
                    get_description(str(output_disease))
                    break
                elif answer_desc_or_prec_intent[0]['intent'] == 'disease_precaution':
                    get_precautions(str(output_disease))
                    break
                elif answer_desc_or_prec_intent[0]['intent'] == 'disease_doctors':
                    get_doctors(str(output_disease))
                    break
                elif answer_desc_or_prec_intent[0]['intent'] == 'reset':
                    print('I will start from the beginning.')
                    return
                else:
                    print('I have not understood, please rephrase.')
            
            break
        elif answer_more_intent[0]['intent'] == 'negative':
            print('Okay.')
            break
        elif answer_more_intent[0]['intent'] == 'reset':
            print('I will start from the beginning.')
            return
        else:
            print('I have detected neither a yes or a no. Please rephrase.')

    

def disease_info_prediction_route(message, description_to_disease, disease_names):

    interest_words = get_interest_words(message)

    for word in interest_words:
        if word in disease_names:
            get_description(word)
            break
        detected_disease = difflib.get_close_matches(word.lower(), disease_names)

        for disease_index in range(len(parsed_disease_names)):
            if word in parsed_disease_names[disease_index]:
                detected_disease.append(disease_names[disease_index])

        detected_disease = list(dict.fromkeys(detected_disease))

        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_description(detected_disease[0])
            break
        elif detected_disease:
            print('Please choose from the following detected diseases:')
            print(detected_disease)
            while(True):
                detected_disease_message = input('')
                detected_disease_message_intent = predict_class(detected_disease_message)

                if detected_disease_message in detected_disease:
                    get_description(detected_disease_message.strip())
                    break
                elif detected_disease_message_intent[0]['intent'] == 'reset':
                    print('I will start from the beginning.')
                    return
                else:
                    print('Please check your spelling.')
            break
        elif interest_words[-1] == word:
            print('Please specify a disease that you want a description for.')
            not_understood_message = input('')
            not_understood_message_intent = predict_class(not_understood_message)
            if not_understood_message_intent[0]['intent'] == 'reset':
                print('I will start from the beginning.')
                return
            disease_info_prediction_route(not_understood_message, description_to_disease, disease_names)

def disease_precautions_prediction_route(message, disease_to_precautions, disease_names):
    
    interest_words = get_interest_words(message)

    for word in interest_words:
        if word in disease_names:
            get_precautions(word)
            break

        detected_disease = difflib.get_close_matches(word.lower(), disease_names)

        for disease_index in range(len(parsed_disease_names)):
            if word in parsed_disease_names[disease_index]:
                detected_disease.append(disease_names[disease_index])

        detected_disease = list(dict.fromkeys(detected_disease))

        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_precautions(detected_disease[0])
            break

        elif detected_disease:
            print('Please choose from the following detected diseases:')
            print(detected_disease)
            while(True):
                detected_disease_message = input('')
                detected_disease_message_intent = predict_class(detected_disease_message)
                
                if detected_disease_message in detected_disease:
                    get_precautions(detected_disease_message.strip())
                    break
                elif detected_disease_message_intent[0]['intent'] == 'reset':
                    print('I will start from the beginning.')
                    return
                else:
                    print('Please check your spelling.')
            break

        elif interest_words[-1] == word:
            print('Please specify a disease that you want precautions for.')
            not_understood_message = input('')
            not_understood_message_intent = predict_class(not_understood_message)
            if not_understood_message_intent[0]['intent'] == 'reset':
                print('I will start from the beginning.')
                return
            disease_precautions_prediction_route(not_understood_message, disease_to_precautions, disease_names)

def disease_doctors_prediction_route(message, disease_names):
    interest_words = get_interest_words(message)

    for word in interest_words:
        if word in disease_names:
            get_doctors(word)
            break

        detected_disease = difflib.get_close_matches(word.lower(), disease_names)

        for disease_index in range(len(parsed_disease_names)):
            if word in parsed_disease_names[disease_index]:
                detected_disease.append(disease_names[disease_index])

        detected_disease = list(dict.fromkeys(detected_disease))

        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_doctors(detected_disease[0])
            break

        elif detected_disease:
            print('Please choose from the following detected diseases:')
            print(detected_disease)
            while(True):
                detected_disease_message = input('')
                detected_disease_message_intent = predict_class(detected_disease_message)
                if detected_disease_message in detected_disease:
                    get_doctors(detected_disease_message.strip())
                    break
                elif detected_disease_message_intent[0]['intent'] == 'reset':
                    print('I will start from the beginning.')
                    return
                else:
                    print('Please check your spelling.')
            break

        elif interest_words[-1] == word:
            print('Please specify a disease that you want to know doctors for.')
            not_understood_message = input('')
            not_understood_message_intent = predict_class(not_understood_message)
            if not_understood_message_intent[0]['intent'] == 'reset':
                print('I will start from the beginning.')
                return
            disease_doctors_prediction_route(not_understood_message, disease_names)

def list_symptoms(unique_symptoms):
    print('These are the symptoms from my database:')
    for symptom in sorted(unique_symptoms):
        print('- ' + str(symptom.replace('_', ' ')) + ';')

def list_diseases(disease_names):
    print('These are the diseases from my database:')
    for disease in sorted(disease_names):
        print('- ' + str(disease.replace('_', ' ')) + ';')

print('RELU GO!')

while True:
    message = input("")
    ints = predict_class(message)
    tag = ints[0]['intent']
    res, probability = get_response(ints, intents)

    if float(probability) < 0.9:
        print('This task is out of my scope.')
        continue

    if tag == 'list_symptoms':
        list_symptoms(unique_symptoms)

    if tag == 'list_diseases':
        list_diseases(disease_names)

    if tag == 'symptom':
        settled_symptoms = []
        symptom_prediction_route(message, unique_symptoms, settled_symptoms, weight_to_symptom)

    if tag == 'disease_info':
        disease_info_prediction_route(message, description_to_disease, disease_names)

    if tag == 'disease_precaution':
        disease_precautions_prediction_route(message, disease_to_precautions, disease_names)

    if tag == 'disease_doctors':
        disease_doctors_prediction_route(message, disease_names)

    if tag == 'goodbye':
        break

    if tag == 'reset':
        print('I will start from the beginning.')

    print(res)
    