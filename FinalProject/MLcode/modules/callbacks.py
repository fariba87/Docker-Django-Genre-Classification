import os
import tensorflow as tf
########## tensorboard ###############################################
root_logdir=  "../data/my_logs"# os.path.join(os.curdir, "../Data/my_logs_ctc")
os.makedirs(root_logdir, exist_ok=True)
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir(root_logdir= root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

########## exponentiallearning rate schedule  ###############################################
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
########## exponentiallearning rate by plateau  ###############################################
lr_schedulerPlat = tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='loss',
                                    factor=0.5,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=0.2,
                                    cooldown=0,
                                    min_lr=0)

########## early stopping callback  ###############################################
earlystopping = tf.keras.callbacks.EarlyStopping(patience=10)
########## checkpoint ###############################################

CHECKPOINT_DIR = "../data/" + 'checkpoint' + "/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
filepath2 = '/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
filepath = os.path.join(CHECKPOINT_DIR , filepath2)
checkpoint =tf.keras.callbacks.ModelCheckpoint(filepath,
                                              verbose=1,
                                              save_best_only=True, monitor ="loss")
########## backup ###############################################
CHECKPOINT_DIR1 = "../data/" + 'checkpointbackup' + "/"
os.makedirs(CHECKPOINT_DIR1, exist_ok=True)
backup_ckpt = tf.keras.callbacks.BackupAndRestore(backup_dir=CHECKPOINT_DIR1)



# backup_ckpt , checkpoint , earlystopping , lr_schedulerPlat , tensorboard_cb







