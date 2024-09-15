# training the CNN part of the RoBERTa/CNN-based model on embeddings of questions from Exam Question Dataset

# import libraries
import numpy as np
import pandas as pd
import keras
import keras_nlp
from sklearn.model_selection import train_test_split

from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, LeakyReLU, UnitNormalization, Reshape
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy
from keras.metrics import F1Score

batch_size = 8
learning_rate = 1e-3
epochs = 72

# build the model
inputs_cnn = Input(shape=(512, 768), batch_size=batch_size)
conv1d_1 = Conv1D(128, 3, activation="leaky_relu")(inputs_cnn)
max_pool = MaxPooling1D()(conv1d_1)
conv1d_2 = Conv1D(64, 3, activation="relu")(max_pool)
gmax_pool = GlobalMaxPooling1D()(conv1d_2)
dense_1 = Dense(32, activation="tanh")(gmax_pool)
drop = Dropout(0.3)(dense_1)
dense_2 = Dense(6, activation="softmax")(drop)

cnn = Model(inputs=inputs_cnn, outputs=dense_2)
cnn.compile(optimizer=RMSprop(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=['accuracy', 'f1_score'])

x = np.load('data/bloom/question.npy')
y = np.load('data/bloom/level.npy')

# training process
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=71, stratify=y)
history = cnn.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# save training history
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# save the model
cnn.save('roberta_cnn.keras')
