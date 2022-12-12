import numpy as np
import pandas as pd
from django.conf import settings
from django.DjangoProject.files.storage import default_storage
from django.shortcuts import render
import tensorflow as tf
import pandas as pd
from utils import clean_description
tf.keras.load_model('')
import pickle
model = pickle.load('model')
vectorizer = pickle.load['vectorizer']
def index(request):
    if request.method =='POST':
        file = request.FILES['txtfile']
        file_name = default_storage.save(file.name , file)
        file_url = default_storage.path(file.name)
        txt = pd.read_csv(file_url)
        txt1 = txt.apply(clean_description)
        seg = vectorizer.texts_to_sequences(txt1)
        max_len =2000
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=max_len)
        x_batch = np.expand_dims(X_padded, axis =0)
        predictions = model.predict(x_batch)
        return render(request, "index.html", {"prediction":predictions})




    else:
        return render(request, "index_html")
    return render(request, "index_html")