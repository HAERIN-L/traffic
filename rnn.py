import pathlib2 as pl2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

def hex2dec1(x):
    if ':' in str(x):
        return str(int(x.replace(':', ''), 16))
    else:
        return x

def hex2dec2(x):
    if '0x' in str(x):
        return int(str(x), 16)
    else:
        return x

features = ["pkSeqID", "stime", "flgs", "proto",
            "saddr", "sport", "daddr", "dport",
            "pkts", "bytes", "state", "ltime",
            "seq", "dur", "mean", "stddev",
            "smac", "dmac", "sum", "min",
            "max", "soui", "doui", "sco",
            "dco", "spkts", "dpkts", "sbytes",
            "dbytes", "rate", "srate", "drate",
            "attack", "category", "subcategory"]

missing_data_features = ["smac", "dmac", "soui", "doui",
                         "sco", "dco", "category", "subcategory"]

'''
ps = pl2.Path("./dataset")

dfs = (
    pd.read_csv(p,
                names=features,
                header=None) for p in ps.glob("*.csv")
)

res = pd.concat(dfs, ignore_index=True)
print(res.head(5))
'''

normal_dfs = []
abnormal_dfs = [] # 1:1
abnormal_dfs_3 = [] # 7
abnormal_dfs_1 = []
for i in range(1, 2):
    dataset_file_name = "./dataset/UNSW_2018_IoT_Botnet_Dataset_" + str(i) + ".csv"
    dataset_buf = pd.read_csv(dataset_file_name, names=features, header=None)
    normal_data_count = len(dataset_buf[dataset_buf["attack"] == 0])
    abnormal_dfs.append(dataset_buf[dataset_buf["attack"] == 1].sample(normal_data_count))
    abnormal_dfs_3.append(dataset_buf[dataset_buf["attack"] == 1].sample(int(normal_data_count * 0.3)))
    abnormal_dfs_1.append(dataset_buf[dataset_buf["attack"] == 1].sample(int(normal_data_count * 0.1)))
    normal_dfs.append(dataset_buf[dataset_buf["attack"] == 0])

normal_dataset = pd.concat(normal_dfs, ignore_index=True)
# print(normal_dataset)
abnormal_dataset = pd.concat(abnormal_dfs, ignore_index=True)
abnormal_dataset_3 = pd.concat(abnormal_dfs_3, ignore_index=True)
abnormal_dataset_1 = pd.concat(abnormal_dfs_1, ignore_index=True)
# print(abnormal_dataset)
dataset = pd.concat([normal_dataset, abnormal_dataset], ignore_index=True)
dataset_3 = pd.concat([normal_dataset, abnormal_dataset_3], ignore_index=True)
dataset_1 = pd.concat([normal_dataset, abnormal_dataset_1], ignore_index=True)

# dataset = pd.read_csv("./dataset/IoT_Botnet_Dataset.csv")

dataset.sort_values(by="pkSeqID")
dataset_3.sort_values(by="pkSeqID")
dataset_1.sort_values(by="pkSeqID")
# print(dataset)

dataset=pd.get_dummies(dataset, columns=['flgs', 'proto', 'state'])
dataset = dataset.drop(missing_data_features, axis=1)
dataset['sport'] = dataset['sport'].fillna(-1)
dataset['sport'] = dataset['sport'].apply(hex2dec2)
dataset['dport'] = dataset['dport'].fillna(-1)
dataset['dport'] = dataset['dport'].apply(hex2dec2)
dataset['saddr'] = dataset['saddr'].apply(lambda x: hex2dec1(x))
dataset['daddr'] = dataset['daddr'].apply(lambda x: hex2dec1(x))
dataset['sport'] = dataset['sport'].apply(hex2dec2)
dataset['dport'] = dataset['dport'].apply(hex2dec2)

dataset['daddr']=dataset['daddr'].str.replace('.','')
dataset['saddr']=dataset['saddr'].str.replace('.','')
dataset['saddr'] = dataset['saddr'].astype('float32')
dataset['daddr'] = dataset['daddr'].astype('float32')

dataset_3=pd.get_dummies(dataset_3, columns=['flgs', 'proto', 'state'])
dataset_3 = dataset_3.drop(missing_data_features, axis=1)
dataset_3['sport'] = dataset_3['sport'].fillna(-1)
dataset_3['sport'] = dataset_3['sport'].apply(hex2dec2)
dataset_3['dport'] = dataset_3['dport'].fillna(-1)
dataset_3['dport'] = dataset_3['dport'].apply(hex2dec2)
dataset_3['saddr'] = dataset_3['saddr'].apply(lambda x: hex2dec1(x))
dataset_3['daddr'] = dataset_3['daddr'].apply(lambda x: hex2dec1(x))
dataset_3['sport'] = dataset_3['sport'].apply(hex2dec2)
dataset_3['dport'] = dataset_3['dport'].apply(hex2dec2)

dataset_3['daddr']=dataset_3['daddr'].str.replace('.','')
dataset_3['saddr']=dataset_3['saddr'].str.replace('.','')
dataset_3['saddr'] = dataset_3['saddr'].astype('float32')
dataset_3['daddr'] = dataset_3['daddr'].astype('float32')

dataset_1=pd.get_dummies(dataset_1, columns=['flgs', 'proto', 'state'])
dataset_1 = dataset_1.drop(missing_data_features, axis=1)
dataset_1['sport'] = dataset_1['sport'].fillna(-1)
dataset_1['sport'] = dataset_1['sport'].apply(hex2dec2)
dataset_1['dport'] = dataset_1['dport'].fillna(-1)
dataset_1['dport'] = dataset_1['dport'].apply(hex2dec2)
dataset_1['saddr'] = dataset_1['saddr'].apply(lambda x: hex2dec1(x))
dataset_1['daddr'] = dataset_1['daddr'].apply(lambda x: hex2dec1(x))
dataset_1['sport'] = dataset_1['sport'].apply(hex2dec2)
dataset_1['dport'] = dataset_1['dport'].apply(hex2dec2)

dataset_1['daddr']=dataset_1['daddr'].str.replace('.','')
dataset_1['saddr']=dataset_1['saddr'].str.replace('.','')
dataset_1['saddr'] = dataset_1['saddr'].astype('float32')
dataset_1['daddr'] = dataset_1['daddr'].astype('float32')

y = []
y.append(dataset[["attack"]])
y.append(dataset_3[["attack"]])
y.append(dataset_1[["attack"]])

x = []
x.append(dataset.drop(["attack"], axis=1))
x.append(dataset_3.drop(["attack"], axis=1))
x.append(dataset_1.drop(["attack"], axis=1))

for i in range(3):
    y[i] = y[i].astype('float32').to_numpy()
    # print(y)

    x[i] = x[i].astype('float32').to_numpy()
    # print(x)

history = []
for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(x[i], y[i], stratify=y[i], test_size=0.2, random_state=42)

    x_train = x_train.reshape((-1, 1, x[i].shape[1]))
    x_test = x_test.reshape((-1, 1, x[i].shape[1]))

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(units=8, input_shape=(1, x[i].shape[1])))
    model.add(keras.layers.Dense(1,activation='sigmoid'))

    model.summary()
    rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    history.append(model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test)))

for i in range(3):
    plt.plot(history[i].history['loss'])
    plt.plot(history[i].history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train(1:1)', 'val(1:1)',
            'train(7:3)', 'val(7:3)',
            'train(9:1)', 'val(9:1)'], loc='upper right')
plt.show()

for i in range(3):
    plt.plot(history[i].history['accuracy'])
    plt.plot(history[i].history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train(1:1)', 'va(1:1)l',
            'train(7:3)', 'val(7:3)',
            'train(9:1)', 'val(9:1)'], loc='lower right')
plt.show()