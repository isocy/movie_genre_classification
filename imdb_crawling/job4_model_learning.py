import os
import json
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

with open('./datasets/variables_in_preprocessing.json', 'r') as file:
    variables = json.load(file)
dataset = np.load('./datasets/dataset.npz')

word_cnt = variables['word_cnt']
token_cnt_max = variables['token_cnt_max']

train_descriptions = dataset['train_descriptions']
test_descriptions = dataset['test_descriptions']
train_categories = dataset['train_categories']
test_categories = dataset['test_categories']

model = Sequential([
    Embedding(word_cnt + 1, word_cnt // 40, input_length=token_cnt_max),
    Conv1D(64, kernel_size=5, padding='same', activation='relu'),
    MaxPooling1D(pool_size=1),
    LSTM(256, activation='tanh', return_sequences=True),
    Dropout(0.3),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.3),
    LSTM(128, activation='tanh'),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(6, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fit_hist = model.fit(train_descriptions, train_categories, batch_size=256, epochs=10,
                     validation_data=(test_descriptions, test_categories))

plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

if not os.path.isdir('./models/'):
    os.mkdir('./models/')
model.save('./models/movie_description_classification_model_{}.h5'.format(round(fit_hist.history['val_accuracy'][-1], 3)))
