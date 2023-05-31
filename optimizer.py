import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import csv_dataset
import model
import matplotlib.pyplot as plt

features = ["pkSeqID", "stime", "flgs", "proto",
            "saddr", "sport", "daddr", "dport",
            "pkts", "bytes", "state", "ltime",
            "seq", "dur", "mean", "stddev",
            "smac", "dmac", "sum", "min",
            "max", "soui", "doui", "sco",
            "dco", "spkts", "dpkts", "sbytes",
            "dbytes", "rate", "srate", "drate",
            "attack", "category", "subcategory"]

dataset_path = ["/Users/haerin/Documents/2023_캡스톤/Ransomatrix-main (2)/dataset/IoT_Botnet_Dataset_70.csv"]

to_category = ['state']
missing_features = ["smac", "dmac", "soui", "doui",
                    "sco", "dco"]
drop_cols = ["category", "subcategory"]

ntds = csv_dataset.NetworkTraffic(dataset_path, to_category)
ntds.set_dataset(header="infer")
ntds.unsw_dataset_preprocessor(to_category=to_category,
                                missing_features=missing_features,
                                drop_cols=drop_cols)
ntds.dataset.sort_values(by="pkSeqID", inplace=True)
ntds.dataset = ntds.dataset[['saddr', 'pkSeqID', 'seq', 'dport',
                            'ltime', 'stime', 'sbytes',
                            'rate', 'bytes', 'daddr', 'pkts', 'attack']]

classifierRNN = model.NetworkTrafficClassifier()
classifierCNN = model.NetworkTrafficClassifier()

# Set up activation function (fixed to sigmoid)
activation_function = "sigmoid"

# Set up optimizer functions to try
optimizers = ["Adam","RMSprop","SGD","Adadelta","Adagrad", "Nadam"]

rnn_results = []
cnn_results = []

for optimizer in optimizers:
    rnn_acc = []
    rnn_precision = []
    rnn_recall = []
    rnn_f1_score = []
    cnn_acc = []
    cnn_precision = []
    cnn_recall = []
    cnn_f1_score = []
    history = []

    for _ in range(3):
        x_train, x_test, y_train, y_test = ntds.make_train_test("pkSeqID", "float32", "attack", balanced=True, test_ratio=0.2)

        # RNN model training and evaluation
        classifierRNN.set_model(model_name="SimpleRNN", units=8, init="HeNormal", input_shape=[1, x_train.shape[1]],
                                activation=[activation_function], learning_rate=1e-4, optimizer=optimizer,
                                loss="binary_crossentropy", epochs=30, batch_size=2)

        history.append(classifierRNN.start_learning(x_train, x_test, y_train, y_test))

        y_pred = classifierRNN.predict(x_test)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)
        rnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
        rnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
        rnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
        rnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))

        # CNN model training and evaluation
        classifierCNN.set_model(model_name="CNN", units=8, init="HeNormal", input_shape=[x_train.shape[1]],
                                activation=[activation_function, activation_function, activation_function], learning_rate=1e-4, optimizer=optimizer,
                                loss="binary_crossentropy", epochs=30, batch_size=2)

        history.append(classifierCNN.start_learning(x_train, x_test, y_train, y_test))

        y_pred = classifierCNN.predict(x_test)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)
        cnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
        cnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
        cnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
        cnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))

    rnn_results.append({
        'Model': 'SimpleRNN',
        'Activation Function': activation_function,
        'Optimizer': optimizer,
        'Accuracy': np.mean(rnn_acc),
        'Precision': np.mean(rnn_precision),
        'Recall': np.mean(rnn_recall),
        'F1 Score': np.mean(rnn_f1_score)
    })

    cnn_results.append({
        'Model': 'CNN',
        'Activation Function': activation_function,
        'Optimizer': optimizer,
        'Accuracy': np.mean(cnn_acc),
        'Precision': np.mean(cnn_precision),
        'Recall': np.mean(cnn_recall),
        'F1 Score': np.mean(cnn_f1_score)
    })

# Print text results
results_df = pd.DataFrame(rnn_results + cnn_results)
print(results_df)

# Set up the data for plotting
rnn_data = {'Accuracy': [result['Accuracy'] for result in rnn_results],
            'Precision': [result['Precision'] for result in rnn_results],
            'Recall': [result['Recall'] for result in rnn_results],
            'F1 Score': [result['F1 Score'] for result in rnn_results]}

cnn_data = {'Accuracy': [result['Accuracy'] for result in cnn_results],
            'Precision': [result['Precision'] for result in cnn_results],
            'Recall': [result['Recall'] for result in cnn_results],
            'F1 Score': [result['F1 Score'] for result in cnn_results]}

data = [rnn_data, cnn_data]
models = ['SimpleRNN', 'CNN']
colors = ['#FF7F50', '#1E90FF']  # Specify colors for RNN and CNN, respectively

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the data
for i, metric in enumerate(data[0].keys()):
    row = i // 2
    col = i % 2
    for j in range(len(models)):
        optimizer_data = [data[j][metric][k] for k in range(len(optimizers))]  # Get the data for the specific model
        x = np.arange(len(optimizers))
        axs[row, col].bar(x + j * 0.2, optimizer_data, width=0.2,
                          label=models[j], color=colors[j])
        axs[row, col].set_title(metric)
        axs[row, col].set_ylim([0, 1])
        axs[row, col].set_xticks(x)
        axs[row, col].set_xticklabels(optimizers)
        axs[row, col].legend()

    # Move the legend to the bottom-right corner
    axs[row, col].legend(loc='lower right')

plt.tight_layout()
plt.show()
