# model.py
from tensorflow import keras


class NetworkTrafficClassifier():
    model_name = ""
    units = 0
    input_shape = []
    activation = ""
    learning_rate = 0.0
    optimizer = ""
    loss = ""
    epochs = 0
    batch_size = 0
    model = 0

    def __set_model_simple_rnn(self):
        '''
        model construction for SimpleRNN
        :return: None
        '''

        if self.init == "GlorotNormal":
            initializer = keras.initializers.GlorotNormal(seed=42)
        elif self.init == "HeNormal":
            initializer = keras.initializers.HeNormal(seed=42)
        else:
            initializer = keras.initializers.HeNormal(seed=42)

        self.model = keras.Sequential()
        self.model.add(keras.layers.SimpleRNN(units=self.units,
                                              kernel_initializer=initializer,
                                              input_shape=tuple(self.input_shape)))
        self.model.add(keras.layers.Dense(1,
                                          kernel_initializer=initializer,
                                          activation=self.activation[0]))

        print(self.model.summary())

    def __set_model_cnn(self):
        '''
        model construction for CNN
        :return: None
        '''

        if self.init == "GlorotNormal":
            initializer = keras.initializers.GlorotNormal(seed=42)
        elif self.init == "HeNormal":
            initializer = keras.initializers.HeNormal(seed=42)
        else:
            initializer = "glorot_uniform"

        self.model = keras.models.Sequential([
            keras.layers.Dense(128,
                               kernel_initializer=initializer,
                               input_dim=self.input_shape[0],
                               activation=self.activation[0]),
            keras.layers.Dense(64,
                               kernel_initializer=initializer,
                               activation=self.activation[1]),
            keras.layers.Dense(1,
                               kernel_initializer=initializer,
                               activation=self.activation[2])
        ])

        print(self.model.summary())

    def set_model(self,
                  model_name,
                  init,
                  units,
                  input_shape,
                  activation,
                  learning_rate,
                  optimizer,
                  loss,
                  epochs,
                  batch_size):
        '''
        set model parameters and construct model
        :param model_name: model name for construction; str type
        :param init: weight initializer; str type
        :param units: units for each layer; int type
        :param input_shape: input shape for learning; list type
        :param activation: activation function; list type
        :param learning_rate: learning rate for learning; float type
        :param optimizer: optimizer function for learning; str type
        :param loss: loss function for learning; str type
        :param epochs: epochs for learning; int type
        :param batch_size: batch size for learning; int type
        :return: None
        '''
        self.model_name = model_name
        self.init = init
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size

        if self.model_name == "SimpleRNN":
            self.__set_model_simple_rnn()
        elif self.model_name == "CNN":
            self.__set_model_cnn()
        else:
            self.__set_model_simple_rnn()

        if self.optimizer == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "RMSprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer == "SGD":
            optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == "Adadelta":
            optimizer = keras.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == "Adagrad":
            optimizer = keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == "Nadam":
            optimizer = keras.optimizers.Nadam(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=["accuracy"])

    def start_learning(self, x_train, x_test, y_train, y_test):
        '''
        start learning by using SimpleRNN
        :param x_train: x train data; ndarray type
        :param x_test: x test data; ndarray type
        :param y_train: y train data; ndarray type
        :param y_test: y test data; ndarray type
        :return: history
        '''
        if self.model_name == "SimpleRNN":
            x_train = x_train.reshape((-1, self.input_shape[0], self.input_shape[1]))
            x_test = x_test.reshape((-1, self.input_shape[0], self.input_shape[1]))
        elif self.model_name == "CNN":
            x_train = x_train.reshape((-1, self.input_shape[0], 1))
            x_test = x_test.reshape((-1, self.input_shape[0], 1))

        return self.model.fit(x_train,
                              y_train,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              validation_data=(x_test, y_test))

    def predict(self, x):
        '''
        Predict class labels for samples in x.
        :param x: Input data; ndarray type.
        :return: y_pred (array-like); class labels for samples in x.
        '''
        if self.model_name == "SimpleRNN":
            x = x.reshape((-1, self.input_shape[0], self.input_shape[1]))
        elif self.model_name == "CNN":
            x = x.reshape((-1, self.input_shape[0], 1))
        y_pred = self.model.predict(x)
        return y_pred

    def model_session_free(self):
        keras.backend.clear_session()
