import tensorflow as tf
tf.random.set_seed(1)







class GenreModel(tf.keras.Model):

    def __init__(self, num_layer, num_units, num_classes, len_vocab, dropout):
        super(GenreModel,self).__init__()
        self.input_layer = tf.keras.layers.Input(shape = (2000,))

        self.embedding = tf.keras.layers.Embedding(input_length=2000,input_dim=len_vocab+1,output_dim=256)
        #input_dim = Vocab_size, output_dim = 8, input_length = max_length)
        self.gru1 = tf.keras.layers.GRU(num_units, return_sequences=True)
        self.droput = tf.keras.layers.Dropout(dropout)
        self.gru2 = tf.keras.layers.GRU(num_units, return_sequences=False)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.num_layer =num_layer
        self.model = None


    def __call__(self):#, x):
        xo = self.embedding(self.input_layer)
        xo = self.gru1(xo)
        xo = self.droput(xo)
        xo = self.gru2(xo)
        xo = self.droput(xo)
        xo = self.classifier(xo)
        model = tf.keras.Model(inputs = self.input_layer , outputs =xo)
        self.model = model
        return self# model #xo

#print(my_model.summary())
#print('len=', len(my_model.trainable_variables))
