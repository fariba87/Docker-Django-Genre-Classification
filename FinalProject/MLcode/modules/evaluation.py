from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
import numpy as np
# prec = tf.keras.metrics.Precison
# rec = tf.keras.metrics.Recall
def calc_confusion_matrix(y_true , y_pred):
    Ypred = np.argmax(y_pred, axis=1)
    Ytrue = np.argmax(y_true, axis=1)
    return confusion_matrix(Ytrue , Ypred)

def calc_precision_recall_f1(y_true , y_pred):
    Ypred = np.argmax(y_pred, axis=1)
    Ytrue = np.argmax(y_true, axis=1)
    precision = precision_score(Ytrue , Ypred, average='macro')
    recall = recall_score(Ytrue , Ypred, average='macro')
    f1score = f1_score(Ytrue , Ypred, average='macro')
    return precision , recall , f1score
calc_loss= tf.keras.losses.CategoricalCrossentropy()

