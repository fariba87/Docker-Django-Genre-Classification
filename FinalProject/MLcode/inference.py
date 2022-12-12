import pickle
import tensorflow as tf
from .utils import clean_description
import numpy as np
import pandas as pd

#le.inverse_transform
# loading
with open('tokenizer.pickle', 'rb') as handle:
    vectorizer= pickle.load(handle)
with open('le.pickle', 'rb') as handle:
    le= pickle.load(handle)


model = tf.keras.Model.load_model('modelcheckpoint')
def inference(file ): #file is a string of words
    txt = pd.read_csv(file)
    txt1 = txt.apply(clean_description)
    seq = vectorizer.texts_to_sequences(txt1)
    max_len = 2000
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=max_len)
    x_batch = np.expand_dims(X_padded, axis=0)
    predictions = model.predict(x_batch)
    Ypred = np.argmax(predictions, axis=1)
    output = le.inverse_transform(Ypred)
    print(output)


