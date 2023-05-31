# main.py
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import csv_dataset
import model

def check_train_test_ratio(dataset_path, features):
    ntds = []
    for ds in dataset_path:
        ntds.append(csv_dataset.NetworkTraffic([ds], features))

    for i in range(3):
        ntds[i].set_dataset(header="infer")
        ntds[i].dataset.sort_values(by=features[0])

    to_category = ['flgs', 'proto', 'state']
    missing_features = ["smac", "dmac", "soui", "doui",
                        "sco", "dco"]
    drop_cols = ["category", "subcategory"]

    for i in range(3):
        ntds[i].unsw_dataset_preprocessor(to_category=to_category,
                                          missing_features=missing_features,
                                          drop_cols=drop_cols)

        x_train, x_test, y_train, y_test = ntds[i].make_train_test(nptype="float32",
                                                                   y_cols="attack")

        print("dataset file: ", dataset_path[i])
        unique, counts = numpy.unique(y_train, return_counts=True)
        print("y_train: ", dict(zip(unique, counts)))
        unique, counts = numpy.unique(y_test, return_counts=True)
        print("y_test: ", dict(zip(unique, counts)))

    # print(x_train)


def print_ratio(dataset_path, features):
    ntds = csv_dataset.NetworkTraffic(dataset_path, features)
    print(ntds.count_data("attack", [0, 1]))

if __name__ == "__main__":
    features = ["pkSeqID", "stime", "flgs", "proto",
                "saddr", "sport", "daddr", "dport",
                "pkts", "bytes", "state", "ltime",
                "seq", "dur", "mean", "stddev",
                "smac", "dmac", "sum", "min",
                "max", "soui", "doui", "sco",
                "dco", "spkts", "dpkts", "sbytes",
                "dbytes", "rate", "srate", "drate",
                "attack", "category", "subcategory"]

    # dataset_path = ["./dataset/UNSW_2018_IoT_Botnet_Dataset_" + str(i) + ".csv" for i in range(1, 75)]
    dataset_path = ["./dataset/IoT_Botnet_Dataset_50.csv",
                    "./dataset/IoT_Botnet_Dataset_70.csv",
                    "./dataset/IoT_Botnet_Dataset_90.csv"]

    to_category = ['flgs', 'proto', 'state']
    missing_features = ["smac", "dmac", "soui", "doui",
                        "sco", "dco"]
    drop_cols = ["category", "subcategory"]

    ntds = []
    ntds.append(csv_dataset.NetworkTraffic([dataset_path[0]], features))
    ntds[0].set_dataset(header="infer")
    ntds[0].unsw_dataset_preprocessor(to_category=to_category,
                                      missing_features=missing_features,
                                      drop_cols=drop_cols)
    ntds[0].dataset = ntds[0].dataset[['saddr', 'seq', 'dport', 'pkSeqID',
                                       'ltime', 'stime', 'sbytes', 'state_ACC',
                                       'rate', 'bytes', 'daddr', 'pkts',
                                       'spkts', 'dur', 'attack']]
    x_train, x_test, y_train, y_test = ntds[0].make_train_test(nptype="float32",
                                                               y_cols="attack")

    ntds.append(csv_dataset.NetworkTraffic([dataset_path[0]], features))
    ntds[1].set_dataset(header="infer")
    ntds[1].dataset.sort_values(by="pkSeqID")
    x = ntds[1].dataset[['pkSeqID', 'spkts', 'pkts', 'sbytes',
                         'dpkts', 'bytes', 'dbytes', 'drate',
                         'srate', 'rate', 'dur', 'stime',
                         'ltime', 'sum']]
    y = ntds[1].dataset["attack"]
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=42)

    scaler = StandardScaler()
    x_train_ = scaler.fit_transform(x_train_)
    x_test_ = scaler.transform(x_test_)

    history = []

    classifierRNN = model.NetworkTrafficClassifier()
    classifierRNN.set_model(model_name="SimpleRNN",
                             units=8,
                             input_shape=[1, x_train.shape[1]],
                             activation=["relu"],
                             learning_rate=1e-4,
                             optimizer="Adam",
                             loss="binary_crossentropy",
                             epochs=30,
                             batch_size=32)
    history.append(classifierRNN.start_learning_simple_rnn(x_train, x_test, y_train, y_test))


    classifierRNN.set_model(model_name="SimpleRNN",
                            units=8,
                            input_shape=[1, x_train_.shape[1]],
                            activation=["relu"],
                            learning_rate=1e-4,
                            optimizer="Adam",
                            loss="binary_crossentropy",
                            epochs=30,
                            batch_size=32)
    history.append(classifierRNN.start_learning_simple_rnn(x_train_, x_test_, y_train_, y_test_))

    # CNN 객체지향 방법로 참고해서 코드 수정
    class CNNClassifier:
        def __init__(self, input_shape=None):
            self.input_shape = input_shape
            self.model = None

        def set_model(self, input_shape=None):
            if input_shape is not None:
                self.input_shape = input_shape
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, input_dim=self.input_shape[1], activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        def start_learning(self, x_train, y_train, epochs, batch_size, validation_data):
            history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
            return history


    classifierCNN = CNNClassifier()
    classifierCNN.set_model(input_shape=(None, x_train.shape[1]))
    history.append(classifierCNN.start_learning(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test)))

    classifierCNN.set_model(input_shape=(None, x_train_.shape[1]))
    history.append(classifierCNN.start_learning(x_train_, y_train_, epochs=30, batch_size=32, validation_data=(x_test_, y_test_)))


    for i in range(4):
        # plt.plot(history[i].history['accuracy'])
        plt.plot(history[i].history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["RNN-RandomForest", "RNN-Correlation",
                "CNN-RandomForest", "CNN-Correlation"], loc='lower right')
    plt.show()
