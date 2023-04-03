# File that trains and stores chatbot learning model

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD
from keras import backend
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()


intents = json.loads(open('json\intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents.get('intents'):
    for pattern in intent.get('patterns'):
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent.get('tag')))
        if intent.get('tag') not in classes:
            classes.append(intent.get('tag'))

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
output_empty = [0]*len(classes)
print(documents)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
#print(training)
training = np.array(training, dtype=object)

x = list(training[:, 0])
y = list(training[:, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle=True, random_state=42,stratify=y)
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

model = Sequential()

model.add(Dense(256, input_shape=(len(x_train[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[rmse,'accuracy','mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])

hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)

print(len(x_test), len(x_test[0]), len(y_test), len(y_test[0]))
y_prediction = model.predict(x_test)
y_prediction = np.argmax (y_prediction, axis = 1)
y_test=np.argmax(y_test, axis=1)
print(y_test)
#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(y_test, y_prediction , normalize='pred')
print(accuracy_score(y_test, y_prediction))
print(classification_report(y_test, y_prediction))
#pyplot.plot(hist.history['accuracy'])

pyplot.plot(hist.history['cosine_proximity'])
pyplot.xlabel('Epochs')
pyplot.ylabel('Cosine proximity')
# pyplot.plot(hist.history['mean_absolute_error'])
# pyplot.plot(hist.history['rmse'])
# pyplot.plot(hist.history['mean_absolute_percentage_error'])
# pyplot.plot(hist.history['cosine_proximity'])
pyplot.show()
df_cm = pd.DataFrame(result, range(11), range(11))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
print(len(result[0]), len(result))


# model.save('chatbotmodel.h5', hist)


print('Done')