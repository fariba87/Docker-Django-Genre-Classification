import tensorflow as tf
import numpy as np
from ConFig.config import ConfigReader
from modules.datareader import DataLoader,train_val_spliter
from modules.model import GenreModel
from modules.evaluation import calc_confusion_matrix, calc_precision_recall_f1, calc_loss
from modules.callbacks import backup_ckpt , checkpoint , earlystopping , lr_schedulerPlat , tensorboard_cb
tf.random.set_seed(1)
import os
absolute_path = os.getcwd()
print('GPU',tf.test.is_gpu_available())
absolute_path = os.path.dirname(__file__)
absolute_path
model_path=os.path.join(absolute_path, 'data/data.csv')
print(model_path)
os.chdir('..')
#from modules.
#from modules.callbacks import callbacks
p = os.getcwd()
data_path = os.path.join(p, 'data/data.csv')#'../data.data.csv'# model_path #'D:/FanapPlus/data/data.csv'#os.path.join(os.getcwd(), '/data/data.csv')
print(data_path)
def get_config_data_model():
    cfg = ConfigReader()#conf_path = data_path )
    xx,yy , len_vocab , tokenizer, le= DataLoader(data_dir =data_path, shuffle=True)
    model1 = GenreModel(num_layer= cfg.NumGRUlayer, num_units = cfg.NumGRUunit, num_classes = 10, len_vocab= len_vocab,  dropout = cfg.dropout)
    model1()
    model = model1.model
    return cfg,xx ,yy, model , model1 , tokenizer, le
cfg,xx, yy , model , model1 , tokenizer , le = get_config_data_model()
## as data is imbalanced, it is better to use train_test_split from sklearn to take it into consideration for splitting

(x_train, y_train),(x_val, y_val) = train_val_spliter(xx, yy , val_split =0.1)
# count_data = len(xx)
# valsplit= cfg.valsplit
# count_val = np.int(valsplit*count_data)
# x_val = xx[:count_val]
# y_val = yy[:count_val]
# x_train = xx[count_val:]
# y_train = yy[count_val:]

import pickle
#le.inverse_transform
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labelencoder.pickle', 'wb') as handl:
    pickle.dump(le, handl, protocol=pickle.HIGHEST_PROTOCOL)
# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#model.build(input_shape=(None, None, 5)


#total_data  = len(xx[0])

#num_val = 0.2*total_data
#idx = np.random.rand(size = num_val , total_data)
#df_val = df[idx]
#df_train = df.drop[idx]
total_step_train = np.int(len(x_train[0])/cfg.batchSize)
def train_one_epoch():
    loss =[]
    for step in range(0,total_step_train, cfg.batchSize):
        x, y =  xx[step:step+ cfg.batchSize], yy[step:step+ cfg.batchSize]#[step*cfg.batchSize :(step+1)*cfg.batchSize], yy[step*cfg.batchSize :(step+1)*cfg.batchSize]
        with tf.GradientTape() as tape:
            logits = model(x)
            lossBatch = calc_loss(y , logits)
            loss.append(lossBatch)
            #accBatch = calc_acc(y, logits)
            cm = calc_confusion_matrix(y, logits)
            p , r, f = calc_precision_recall_f1(y, logits)
        grads = tape.gradient(lossBatch, model.trainable_variables)
        model1.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        #train_acc_metric.update_state(y, logits)
        if step%200 ==0 :
            print('Training loss(for one  batch) at step %d  is %.4f'%(step, float(lossBatch)))
  #  train_acc = train_acc_metric.result()
  #  train_acc_metric.reset_states()
    return loss
total_step_valid = np.int(len(x_val[0])/cfg.batchSize)
def val_one_epoch():
    for step in range(0,total_step_valid, cfg.batchSize):
        x, y = x_val[step:step+ cfg.batchSize], y_val[step:step+ cfg.batchSize]#*cfg.batchSize :(step+1)*cfg.batchSize], y_val[step*cfg.batchSize :(step+1)*cfg.batchSize]
        logits = model(x)
        val_loss = calc_loss(y, logits)
        cm = calc_confusion_matrix(y, logits)
        p, r, f = calc_precision_recall_f1(y, logits)
 #       val_acc_metric.update_state(y, val_logits)
 #   val_acc = val_acc_metric.result()
 #   val_acc_metric.reset_states()
    return val_loss, p , r, f

#from modules.callbacks import earlystopping, lr_scheduler, tensorboard_cb , checkpoint, backup_ckpt
_callbacks = [backup_ckpt , checkpoint , earlystopping , lr_schedulerPlat , tensorboard_cb]
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True, model=model)
logs={}
loss_epochs=[]
val_loss_epochs=[]
for epoch in range(cfg.TotalEpoch):
    callbacks.on_epoch_begin(epoch, logs=logs)
    loss  = train_one_epoch()
    val_loss, p, r, f = val_one_epoch()
    precision = np.mean(p)
    recall = np.mean(r)
    f1score = np.mean(f)
    loss_mean = np.mean(loss)
    loss_epochs.append(loss_mean)
    val_loss_mean = np.mean(val_loss)
    val_loss_epochs.append(val_loss_mean)

    print('loss and val loss in epoch {} is {} and {} '.format(epoch + 1, loss_mean ,val_loss_mean))
    print('validation metric in epoch {}: precision = {} Recall = {} F1score ={}'.format(epoch + 1, precision, recall, f1score))
  #  np.mean(acc)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(loss_epochs)
plt.title('train loss')
plt.subplot(1,2,2)
plt.plot(val_loss_epochs)
plt.title('validation loss')
plt.show()


def keras_model_training(text_model, X_padded_seq, YY):
    text_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # change loss and acc
    text_model.summary()
    text_model.fit(X_padded_seq, YY,validation_split= 0.1, epochs=cfg.TotalEpoch, batch_size=cfg.batchSize)


