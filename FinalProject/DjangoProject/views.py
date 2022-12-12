from django.shortcuts import render 
# def home(request):
#     return render(request, 'index.html')
import numpy as np
import pandas as pd
from django.conf import settings
from django.DjangoProject.files.storage import default_storage
from django.shortcuts import render
import tensorflow as tf
import pandas as pd
from MLcode.utils import clean_description
from data.checkpoint import model  # model from checkpoint
model = tf.keras.load_model('model.pth') # from checkpoint
import pickle
model = pickle.load('model.pth')
vectorizer = pickle.load['tokenizer.pickle']
def index(request):
    if request.method =='POST':
        file = request.FILES['txtfile']
        file_name = default_storage.save(file.name , file)
        file_url = default_storage.path(file.name)
        txt = pd.read_csv(file_url)
        txt1 = txt.apply(clean_description)
        seq = vectorizer.texts_to_sequences(txt1)
        max_len =2000
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=max_len)
        x_batch = np.expand_dims(X_padded, axis =0)
        predictions = model.predict(x_batch)
        return render(request, "index.html", {"predictions":predictions})




    else:
        return render(request, "index_html")
    return render(request, "index_html")