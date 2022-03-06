import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import load_model
import tensorflow

import json
import random



def response(a):

    loadedEntityCV = pk.load(open('saved_state/EntityCountVectorizer.sav', 'rb'))
    loadedEntityClassifier = pk.load(open('saved_state/entity_model.sav', 'rb'))


    with open('datasets/intents.json') as json_data:
        intents = json.load(json_data)

# Load model to predict user result
    loadedIntentClassifier = load_model('saved_state/intent_model.h5')
    loaded_intent_CV = pk.load(open('saved_state/IntentCountVectorizer.sav', 'rb'))    

    USER_INTENT = ""
    intent_label_map={'Intent': 0, 'askinghelp': 1, 'bajracondition': 2, 'bajraweather': 3, 'buy': 4, 'cottoncondition': 5, 'cottonweather': 6, 'enquireaboutday': 7, 'family': 8, 'farmerpmkisan': 9, 'farmerscheme': 10, 'goodbye': 11, 'greeting': 12, 'groundnutcondition': 13, 'groundnutweather': 14, 'hours': 15, 'maizecondition': 16, 'maizeweather': 17, 'marketprice': 18, 'opentoday': 19, 'outofscope': 20, 'ricecondition': 21, 'riceweather': 22, 'sarcasm': 23, 'sell': 24, 'sugercanecondition': 25, 'sugercaneweather': 26, 'thanks': 27, 'wassup': 28, 'wellness': 29, 'wheatcondition': 30, 'wheatweather': 31}


    user_query = a
     
    query = re.sub('[^a-zA-Z]', ' ', user_query)

     # Tokenize sentence
    query = query.split(' ')

     # Lemmatizing
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]

     # Recreate the sentence from tokens
    processed_text = ' '.join(tokenized_query)
     
     # Transform the query using the CountVectorizer
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()

     # Make the prediction
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
     #     print(predicted_Intent)
    result = np.argmax(predicted_Intent, axis=1)
     
    for key, value in intent_label_map.items():

        if value==result[0]:

               #print(key)
            USER_INTENT = key
            break
          
    for i in intents['intents']:

        if i['tag'] == USER_INTENT:

               
            print(random.choice(i['responses']))
            return(random.choice(i['responses']))


