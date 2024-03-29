# Main execution file that runs the chatbot application

import pickle
import json
import pandas as pd
import spacy
from collections import Counter
from string import punctuation
import difflib
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from tkinter import *
from tkinter import ttk

from disease_training import training_and_data_parsing
from chatbot_init import clean_up_sentence, bag_of_words, predict_class, get_response


# Load needed variables and model from training function
randomFC, model_symptom_list, symptom_to_severity, \
      disease_to_symptoms, description_to_disease, disease_to_precautions, \
        doctors_json_db, parsed_disease_names, unique_symptoms, disease_names = training_and_data_parsing()


# Chatbot parameters initialization:
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('json\intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbotmodel.h5')


# Helper functions for the main tag functions

def get_interest_words(message):
    interest_words = []
    nlp = spacy.load("en_core_web_sm")

    def get_hotwords(text):
        result = []
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB'] 
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

def get_description(disease):
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: I will provide a description for ' + str(disease) + '.')
    txt.insert(END, "\n" + description_to_disease[disease.lower()]['Description'])
    txt.configure(state=DISABLED)
    txt.see(END)

def get_precautions(disease):
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: I will provide precautions for ' + str(disease) + '.')

    for precaution in disease_to_precautions[disease.lower()]:  
        txt.insert(END, "\n" + '- ' + precaution + ';')
    txt.configure(state=DISABLED)
    txt.see(END)

def get_doctors(disease):
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: I will provide proffessional help regarding ' + str(disease) + '.')
    
    for hospital, value in doctors_json_db.items():
        for department, doctors in value.get('departments').items():
            if disease.lower().replace(' ', '') in [x.lower().replace(' ', '') for x in doctors.get('diseases')]:
                txt.insert(END, "\n" + 'Help can be found at ' + hospital + '. ' + 'Link: ' + value['link'])
                txt.insert(END, "\n" + 'It has various departments, but the one of your interest will be: ' + department + '.')
                doctor_list = doctors.get('doctors')
                appointment_list = doctors.get('appointments')
                txt.insert(END, "\n" + 'The names of the doctors that can help you here are:')
                for i in range(len(doctor_list)):  
                    txt.insert(END, "\n" + f'- {doctor_list[i]} ({appointment_list[i]})')
    txt.configure(state=DISABLED)
    txt.see(END)

def get_symptoms(disease):
    symptoms = disease_to_symptoms[disease.lower()]
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: I will provide the symptoms for ' + str(disease) + '.')
    for symptom in symptoms:  
        if not pd.isna(symptom):
            txt.insert(END, "\n" + '- ' + str(symptom).replace('_', ' ') + ';')
    txt.configure(state=DISABLED)
    txt.see(END)


# Main tag functions that execute on tag detection
def list_symptoms(unique_symptoms):
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: These are the symptoms from my database:')
    for symptom in sorted(unique_symptoms):
        txt.insert(END, "\n" + '- ' + str(symptom.replace('_', ' ')) + ';')
    txt.configure(state=DISABLED)
    txt.see(END)

def list_diseases(disease_names):
    txt.configure(state=NORMAL)
    txt.insert(END, "\n\n" + 'Medline: These are the diseases from my database:')
    for disease in sorted(disease_names):
        txt.insert(END, "\n" + '- ' + str(disease.replace('_', ' ')) + ';')
    txt.configure(state=DISABLED)
    txt.see(END)

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

        k=0
        for disease in detected_disease:
            if disease.lower() in message.lower():
                get_doctors(disease)
                k=1
        if k:
            break
        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_doctors(detected_disease[0])
            break

        elif detected_disease:
            
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please choose from the following detected diseases:')
            for disease in detected_disease:
                txt.insert(END, "\n" + f"- {disease}")
            txt.configure(state=DISABLED)
            txt.see(END)
            break

        elif interest_words[-1] == word:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please specify a disease that you want to know doctors for.')
            txt.configure(state=DISABLED)
            txt.see(END)

def disease_precautions_prediction_route(message, disease_names):
    
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

        k=0
        for disease in detected_disease:
            if disease.lower() in message.lower():
                get_precautions(disease)
                k=1
        if k:
            break
        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_precautions(detected_disease[0])
            break

        elif detected_disease:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please choose from the following detected diseases:')
            for disease in detected_disease:
                txt.insert(END, "\n" + f"- {disease}")
            txt.configure(state=DISABLED)
            txt.see(END)
            break

        elif interest_words[-1] == word:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please specify a disease that you want precautions for.')
            txt.configure(state=DISABLED)
            txt.see(END)

def disease_info_prediction_route(message, disease_names):

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

        k=0
        for disease in detected_disease:
            if disease.lower() in message.lower():
                get_description(disease)
                k=1
        if k:
            break
        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_description(detected_disease[0])
            break
        elif detected_disease:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + f"Medline: Please choose from the following detected diseases:")
            for disease in detected_disease:
                txt.insert(END, "\n" + f"- {disease}")
            txt.configure(state=DISABLED)
            txt.see(END)
            break
        elif interest_words[-1] == word:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please specify a disease that you want a description for.')
            txt.configure(state=DISABLED)
            txt.see(END)

def symptom_prediction_route(message, unique_symptoms, settled_symptoms):

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
            symptom_words = symptom.split(' ')
            matching = difflib.get_close_matches(word, symptom_words)  
            if matching:
                detected_symptoms.append(symptom)
        if found_symptoms:
            detected_symptoms.extend(found_symptoms)

    detected_symptoms = list( dict.fromkeys(detected_symptoms) )
    settled_symptoms = list( dict.fromkeys(settled_symptoms) )

    if settled_symptoms:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n\n" + f"Medline: I have understood the following symptoms: {', '.join([symptom.replace('_', ' ') for symptom in settled_symptoms])} and recorded them")
        txt.configure(state=DISABLED)
        txt.see(END)
    else:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n" + "I have not recorded any symptoms.")
        txt.configure(state=DISABLED)
        txt.see(END)

    if detected_symptoms:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n" + "There might be some symptoms I have not understood.")

        for symptom in settled_symptoms:
            if symptom in detected_symptoms:
                detected_symptoms.remove(symptom)

        detected_symptoms_string = ', '.join([symptom.replace('_', ' ') for symptom in detected_symptoms])
        
        txt.insert(END, "\n" + "Symptoms you might have reffered to: " + detected_symptoms_string + '.')
        txt.insert(END, "\n" + "If you reffered to some of the symptoms above, please send me a message that contains these symptoms and the ones I recorded before.")
        txt.configure(state=DISABLED)
        txt.see(END)

    severity = 0
    for symptom in settled_symptoms:
        
        severity += symptom_to_severity[symptom]['weight']


    symptom_presence = []
    for symptom in model_symptom_list:
        if symptom in settled_symptoms:
            symptom_presence.append(1)
        else:
            symptom_presence.append(0)

    qw=pd.DataFrame([symptom_presence],columns=model_symptom_list)
    no_of_symptoms = len(settled_symptoms)

    output = randomFC.predict_proba(qw)
    prediction_classes = randomFC.classes_
    n = 3

    output_disease = np.argsort(output)[:,:-n-1:-1]
    response_diseases = []
    for index in output_disease[0]:
        response_diseases.append(str(prediction_classes[index]))

    if not no_of_symptoms:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n" + "I cannot predict a disease with no symptoms.")
        txt.configure(state=DISABLED)
        txt.see(END)
        return
    else:    
        txt.configure(state=NORMAL)
        txt.insert(END, "\n" + "Possible diseases, ordered by likelihood: " + ', '.join([str(disease) for disease in response_diseases]) + '.')
        if severity >= 13:
            txt.insert(END, "\n" + "You should consult a doctor as soon as possible.")
        else:
            txt.insert(END, "\n" + "You should go to a doctor when you have the time.")
        txt.configure(state=DISABLED)
        txt.see(END)

    if no_of_symptoms < 3:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n" + "I am not very sure about this prediction as the number of symptoms given is low.")
        txt.configure(state=DISABLED)
        txt.see(END)

def disease_to_symptoms_prediction_route(message, disease_names):
    interest_words = get_interest_words(message)

    for word in interest_words:
        if word in disease_names:
            get_symptoms(word.lower())
            break
        detected_disease = difflib.get_close_matches(word.lower(), disease_names)

        for disease_index in range(len(parsed_disease_names)):
            if word in parsed_disease_names[disease_index]:
                detected_disease.append(disease_names[disease_index])

        detected_disease = list(dict.fromkeys(detected_disease))

        k=0
        for disease in detected_disease:
            if disease.lower() in message.lower():
                get_symptoms(disease.lower())
                k=1
        if k:
            break
        if detected_disease and len(detected_disease) == 1 and detected_disease[0] in disease_names:
            get_symptoms(detected_disease[0].lower())
            break
        elif detected_disease:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + f"Medline: Please choose from the following detected diseases:")
            for disease in detected_disease:
                txt.insert(END, "\n" + f"- {disease}")
            txt.configure(state=DISABLED)
            txt.see(END)
            break
        elif interest_words[-1] == word:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: Please specify a disease that you want symptoms for.')
            txt.configure(state=DISABLED)
            txt.see(END)


# Creating tkinter graphical user interface

# Execute when send button is pressed
def send():
    send = '\n' + "You: " + e.get()
    txt.configure(state=NORMAL)
    txt.insert(END, "\n" + send)
    txt.configure(state=DISABLED)
    txt.see(END)

    message = e.get().lower()

    e.delete(0, END)

    ints = predict_class(message)
    tag = ints[0]['intent']
    res, probability = get_response(ints, intents)

    if float(probability) < 0.75:
        txt.configure(state=NORMAL)
        txt.insert(END, "\n\n" + "Medline: This task is out of my scope.")
        txt.configure(state=DISABLED)
        txt.see(END)
    else:
        if tag == 'list_symptoms':
            list_symptoms(unique_symptoms)

        if tag == 'list_diseases':
            list_diseases(disease_names)
            
        if tag == 'symptom':
            settled_symptoms = []
            symptom_prediction_route(message, unique_symptoms, settled_symptoms)

        if tag == 'disease_info':
            disease_info_prediction_route(message, disease_names)

        if tag == 'disease_precaution':
            disease_precautions_prediction_route(message, disease_names)
            
        if tag == 'disease_doctors':
            disease_doctors_prediction_route(message, disease_names)

        if tag == 'disease_to_symptoms':
            disease_to_symptoms_prediction_route(message, disease_names)
    
        if res:
            txt.configure(state=NORMAL)
            txt.insert(END, "\n\n" + 'Medline: ' + res) 
            txt.configure(state=DISABLED)
            txt.see(END)


# Defining tkinter interface parameters
help_info = "Here is some information I can provide: \n" + \
    "- predicting a disease based on your symptoms;\n" + \
    "- description of a disease;\n -what symptoms a disease has;\n" + \
    " - doctors and hospitals for a disease;\n - precautions/basic treatment for a disease;\n" + \
    "- list symptoms I know;\n - list diseases I know. "

root = Tk()
root.title("Chatbot")
 
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
 
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
def open_popup():
   top= Toplevel(root)
   top.grab_set()
   top.geometry("420x220")
   top.title("Help Window")
   Label(top, text= help_info, font=FONT_BOLD, fg=TEXT_COLOR, bg=BG_COLOR, pady=10, anchor="w").place(relwidth=1, relheight=1)
   top.resizable(width=False, height=False)
   

root.title("Medline")
root.resizable(width=False, height=False)
root.configure(width=1500, height=800, bg=BG_COLOR)

# head label
label = Label(root, bg=BG_COLOR, fg=TEXT_COLOR,
                    text="Medline - Medical Chatbot", font=(FONT_BOLD,20), pady=10, anchor="w",)
label.place(relwidth=0.78, relx=0.02)


help_button = Button(root, text= "What can you ask me?", font=(FONT_BOLD, 12), bg=BG_GRAY, command= open_popup)
help_button.place(relx=0.84, rely=0.008, relheight=0.06, relwidth=0.15)


# tiny divider
line = Label(root, width=450, bg=BG_GRAY)
line.place(relwidth=1, rely=0.07, relheight=0.012)

# text widget
txt = Text(root, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                        font=FONT, padx=5, pady=5)
txt.place(relheight=0.745, relwidth=1, rely=0.08)
txt.configure(cursor="arrow", state=DISABLED)

# scroll bar
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)
scrollbar.configure(command=txt.yview)

# bottom label
bottom_label = Label(root, bg=BG_GRAY, height=80)
bottom_label.place(relwidth=1, rely=0.825)

# message entry box
e = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=(FONT, 18))
e.place(relwidth=0.85, relheight=0.1, rely=0.008, relx=0.011)
e.focus()

send = Button(bottom_label, text="Send", font=(FONT_BOLD,22), width=18, bg=BG_GRAY,
                             command=send)
send.place(relx=0.868, rely=0.03, relheight=0.06, relwidth=0.12)

# Make fullscreen
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
 # Run tkinter interface loop
root.mainloop()