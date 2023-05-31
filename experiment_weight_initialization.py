import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv_dataset
import model

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
dataset_path = ["./dataset/IoT_Botnet_Dataset_90.csv", # fixed test dataset
                "./dataset/IoT_Botnet_Dataset_50.csv"]


to_category = ['flgs', 'proto', 'state']
missing_features = ["smac", "dmac", "soui", "doui",
                    "sco", "dco"]
drop_cols = ["category", "subcategory"]

# selected features by using correlation matrix
selected_features_corr = ["pkSeqID", "spkts", "pkts", "sbytes",
                          "dpkts", "bytes", "dbytes", "drate",
                          "srate", "rate", "dur", "stime",
                          "ltime", "sum", "attack"]

# network traffic dataset (UNSW IoT-Bot Dataset)
ntds = csv_dataset.NetworkTraffic([dataset_path[0]],
                                  features)
ntds.set_dataset(header="infer")
ntds.unsw_dataset_preprocessor(to_category=to_category,
                               missing_features=missing_features,
                               drop_cols=drop_cols)
ntds.dataset = ntds.dataset[selected_features_corr]
x_train, x_test, y_train, y_test = ntds.make_train_test(sort_col="pkSeqID",
                                                        nptype="float32",
                                                        y_cols="attack")

ntds_train = csv_dataset.NetworkTraffic([dataset_path[1]],
                                        features)
ntds_train.set_dataset(header="infer")
ntds_train.unsw_dataset_preprocessor(to_category=to_category,
                                     missing_features=missing_features,
                                     drop_cols=drop_cols)
ntds_train.dataset = ntds_train.dataset[selected_features_corr]
x_train, x_dummy, y_train, y_dummy = ntds.make_train_test(sort_col="pkSeqID",
                                                          nptype="float32",
                                                          y_cols="attack")

classifierRNN = model.NetworkTrafficClassifier()

history = []
for i in range(10):
    classifierRNN.set_model(model_name="SimpleRNN",
                                units=8,
                                init="HeNormal",
                                input_shape=[1, x_train.shape[1]],
                                activation=["relu"],
                                learning_rate=1e-4,
                                optimizer="Adam",
                                loss="binary_crossentropy",
                                epochs=30,
                                batch_size=32)

    history.append(classifierRNN.start_learning(x_train, x_test, y_train, y_test))

    # plt.plot(history[i].history['accuracy'])
    plt.plot(history[i].history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(["HeNoraml - RNN"], loc='lower right')
plt.show()