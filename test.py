import numpy as np
import pandas as pd
import sklearn.metrics as metrics
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

dataset_path = [
    "/Users/haerin/Downloads/Ransomatrix-main (2)/dataset/IoT_Botnet_Dataset_90.csv"]

to_category = ['state']
missing_features = ["smac", "dmac", "soui", "doui",
                    "sco", "dco"]
drop_cols = ["category", "subcategory"]

ntds = []
for i in range(4):
    ntds.append(csv_dataset.NetworkTraffic([dataset_path[i]], to_category))

for i in range(4):
    ntds[i].set_dataset(header="infer")
    ntds[i].unsw_dataset_preprocessor(to_category=to_category,
                                      missing_features=missing_features,
                                      drop_cols=drop_cols)
    ntds[i].dataset.sort_values(by="pkSeqID", inplace=True)
    ntds[i].dataset = ntds[i].dataset[['saddr', 'pkSeqID', 'seq', 'dport', 
                                       'ltime', 'stime',  'sbytes',
                                       'rate', 'bytes', 'daddr', 'pkts', 'attack']]

classifierRNN = model.NetworkTrafficClassifier()
classifierCNN = model.NetworkTrafficClassifier()

rnn_acc = []
rnn_precision = []
rnn_recall = []
rnn_f1_score = []
cnn_acc = []
cnn_precision = []
cnn_recall = []
cnn_f1_score = []
history = []

# IoT_Botnet_Dataset_50.csv
x_train, x_test, y_train, y_test = ntds[0].make_train_test("pkSeqID", "float32", "attack", balanced=True, test_ratio=0.2)

classifierRNN.set_model(model_name="SimpleRNN", units=8, init="HeNormal", input_shape=[1, x_train.shape[1]],
                        activation=["relu"], learning_rate=1e-4, optimizer="Adam",
                        loss="binary_crossentropy", epochs=30, batch_size=32)

history.append(classifierRNN.start_learning(x_train, x_test, y_train, y_test))

y_pred = classifierRNN.predict(x_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
rnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
rnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
rnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
rnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))

classifierCNN.set_model(model_name="CNN", units=8, init="HeNormal", input_shape=[x_train.shape[1]],
                        activation=["relu", "relu", "sigmoid"], learning_rate=1e-4, optimizer="Adam",
                        loss="binary_crossentropy", epochs=30, batch_size=32)

history.append(classifierCNN.start_learning(x_train, x_test, y_train, y_test))

y_pred = classifierCNN.predict(x_test)
y_pred_binary = np.where(y_pred > 0.5, 1, 0)
cnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
cnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
cnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
cnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))


import matplotlib.pyplot as plt

# Set up the data for plotting
data = {'Accuracy': [rnn_acc, cnn_acc],
        'Precision': [rnn_precision, cnn_precision],
        'Recall': [rnn_recall, cnn_recall],
        'F1 Score': [rnn_f1_score, cnn_f1_score]}
models = ['SimpleRNN', 'CNN']

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the data
for i, metric in enumerate(data.keys()):
    row = i // 2
    col = i % 2
    metric_data = np.array(data[metric])
    axs[row, col].bar(models, metric_data[:, 0], label='IoT_Botnet_Dataset_70')
    print(metric_data.shape)
    axs[row, col].set_title(metric)
    axs[row, col].set_ylim([0, 1])
    axs[row, col].legend()


plt.tight_layout()
plt.show()
