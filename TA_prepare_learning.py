# the code generates RoBERTa embeddings from the questions in Exam Question Dataset

# import libraries
import numpy as np
import pandas as pd
import keras
import keras_nlp
from sklearn.model_selection import train_test_split
from keras import Input, Model

# prepare RoBERTa from preset
preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset("roberta_base_en")
backbone = keras_nlp.models.RobertaBackbone.from_preset("roberta_base_en")
backbone.trainable = False

# build an encoding model with RoBERTa backbone
inputs = Input(shape=(1,), dtype="string", name="sentence")
preprocess = preprocessor(inputs)
embed = backbone(preprocess)
encoder = Model(inputs=inputs, outputs=embed)

# encode questions
bloom = pd.read_csv('data/bloom/transformed.csv', sep=';')
x = encoder.predict(bloom.Question)
y = np.array(pd.get_dummies(bloom.Level, dtype=int))

# save questions and levels in separate files
np.save('data/bloom/question.npy', x)
np.save('data/bloom/level.npy', y)
