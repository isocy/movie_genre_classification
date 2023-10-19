import os
import json
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df_description_category = pd.read_csv('./crawling_data/all_movie_descriptions.csv')

description_series = df_description_category['description']
category_series = df_description_category['category']

label_encoder = LabelEncoder()
labeled_category = label_encoder.fit_transform(category_series)
label = label_encoder.classes_

oneHotEncoded_category = to_categorical(labeled_category)

user_name = os.getlogin()
if not os.path.isdir(f'C:/Users/{user_name}/AppData/Roaming/nltk_data/tokenizers/punkt/'):
    nltk.download('punkt')
if not os.path.isdir(f'C:/Users/{user_name}/AppData/Roaming/nltk_data/corpora/stopwords/'):
    nltk.download('stopwords')

raw_stopwords = stopwords.words('english')
stopwords = set()
for stopword in raw_stopwords:
    if '\'' in stopword:
        stopwords.update(stopword.split('\''))
    else:
        stopwords.add(stopword)

for description_idx in range(len(description_series)):
    description_series[description_idx] = word_tokenize(description_series[description_idx].lower())

    words = []
    for word_idx in range(len(description_series[description_idx])):
        word = description_series[description_idx][word_idx]
        if word not in stopwords:
            words.append(word)
    description_series[description_idx] = ' '.join(words)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(description_series)
tokenized_descriptions = tokenizer.texts_to_sequences(description_series)
word_cnt = len(tokenizer.index_word)

token_cnt_max = 0
for tokenized_description in tokenized_descriptions:
    if len(tokenized_description) > token_cnt_max:
        token_cnt_max = len(tokenized_description)
tokenized_descriptions = pad_sequences(tokenized_descriptions, token_cnt_max)

train_descriptions, test_descriptions, train_categories, test_categories = train_test_split(
    tokenized_descriptions, oneHotEncoded_category, test_size=0.2)

if not os.path.isdir('./datasets/'):
    os.mkdir('./datasets/')

variables = {'word_cnt': word_cnt, 'token_cnt_max': token_cnt_max}
with open('./datasets/variables_in_preprocessing.json', 'w') as file:
    json.dump(variables, file)
np.save('./datasets/label.npy', label)
np.save('./datasets/tokenized_descriptions.npy', tokenized_descriptions)
np.savez('./datasets/dataset.npz', train_descriptions=train_descriptions, test_descriptions=test_descriptions,
         train_categories=train_categories, test_categories=test_categories)
