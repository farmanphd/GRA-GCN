# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:25:26 2018

@author: Khanh Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:08:49 2018

@author: Khanh Lee
"""

import numpy
# fix random seed for reproducibility

seed = 237
numpy.random.seed(seed)
from sklearn.metrics import accuracy_score, \
    log_loss, \
    classification_report, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    auc, \
    roc_curve, f1_score, recall_score, matthews_corrcoef, auc



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, TimeDistributed, Input, Reshape
from tensorflow.keras.models import Model

# Load the protein data from the CSV file
data = pd.read_csv('RMP-PSSM-Ingio-train.csv')

# Split the data into input sequences and labels
sequences = dataset1[:,1:401].reshape(len(dataset1),20,20,1)
Y1 = dataset1[:,0]
labels = numpy.asarray(Y1)

# Perform label encoding on the protein labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize the input protein sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

# Pad the sequences to ensure equal length
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Reshape the sequences to 2D for 2D-GRU
sequences = np.reshape(sequences, (sequences.shape[0], sequences.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define the 2D-GRU model
input_shape = (X_train.shape[1], X_train.shape[2])
inputs = Input(shape=input_shape)
gru_out = GRU(128, return_sequences=True)(inputs)
gru_out = TimeDistributed(Dense(64, activation='relu'))(gru_out)
flatten = Reshape((-1, 64))(gru_out)
output = Dense(1, activation='sigmoid')(flatten)

model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')