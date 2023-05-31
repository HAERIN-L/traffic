import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
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

dataset_path = ["/Users/haerin/Downloads/Ransomatrix-main (2)/dataset/IoT_Botnet_Dataset_50.csv"]

to_category = ['state']
missing_features = ["smac", "dmac", "soui", "doui",  # 결측된 컬럼 목록
                    "sco", "dco"]
drop_cols = ["category", "subcategory"]  # 제거할 컬럼 목록

ntds = csv_dataset.NetworkTraffic(dataset_path, to_category)    # NetworkTraffic 인스턴스화
                                                                # 경로의 데이터 셋을 사용하여 카테고리로 변환(=to_category)
ntds.set_dataset(header="infer")
ntds.unsw_dataset_preprocessor(to_category=to_category,              # unsw_dataset_preprocessor 호출하여 데이터셋 전처리
                                missing_features=missing_features,
                                drop_cols=drop_cols)
ntds.dataset.sort_values(by="pkSeqID", inplace=True)                # 'pkSeqID' 열 기준으로 정렬

ntds.dataset = ntds.dataset[['saddr', 'pkSeqID', 'seq', 'dport',    # 나열 순서대로 컬럼 선택하고 ntds.dataset에 할당
                             'ltime', 'stime', 'sbytes',
                             'rate', 'bytes', 'daddr', 'pkts', 'attack']]

classifierRNN = model.NetworkTrafficClassifier() # NetworkTrafficClassifier 클래스를 인스턴스화하여 
classifierCNN = model.NetworkTrafficClassifier() # 세 개의 분류기 객체를 생성

# 사용할 손실 함수들을 나타내는 리스트
loss_functions = ['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error']
models = ['SimpleRNN', 'CNN']

#  정확도, 정밀도, 재현율, F1 스코어를 기록하기 위한 리스트
rnn_acc = []
rnn_precision = []
rnn_recall = []
rnn_f1_score = []
cnn_acc = []
cnn_precision = []
cnn_recall = []
cnn_f1_score = []
history = []  # 학습 과정에서 발생한 로스 값과 정확도 값을 저장하기 위한 리스트

# ntds.make_train_test 사용하여 학습용, 테스트용 데이터 생성
# "pkSeqID"를 기준으로 분리, 타입을 "float32" 지정, "attack" 타겟 변수로 사용
x_train, x_test, y_train, y_test = ntds.make_train_test("pkSeqID", "float32", "attack", balanced=True, test_ratio=0.2)

# 정수형으로 변환하고 범위를 확인
y_train = y_train.astype(int)
y_test = y_test.astype(int)
print("y_train range:", np.min(y_train), np.max(y_train))
print("y_test range:", np.min(y_test), np.max(y_test))

# 1보다 큰 값인 경우 1로 수정 -> 이진 분류로 변환하는 작업
y_train = np.where(y_train > 1, 1, y_train)
y_test = np.where(y_test > 1, 1, y_test)
print("Modified y_train range:", np.min(y_train), np.max(y_train))
print("Modified y_test range:", np.min(y_test), np.max(y_test))

# loss_func : loss_functions 리스트에 있는 손실함수 차례로 가져옴 
for loss_func in loss_functions:

    # RNN 모델, 손실 함수로 loss_func 변수에 저장된 값을 사용
    classifierRNN.set_model(model_name="SimpleRNN", units=8, init="HeNormal", input_shape=[1, x_train.shape[1]],
                            activation=["relu"], learning_rate=1e-4, optimizer="Adam",
                            loss=loss_func, epochs=30, batch_size=32)

    history.append(classifierRNN.start_learning(x_train, x_test, y_train, y_test))

    # classifierRNN을 사용하여 테스트 데이터에 대한 예측 수행
    y_pred = classifierRNN.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)    
    rnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
    rnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
    rnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
    rnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))

    # CNN 모델, 손실 함수로 loss_func 변수에 저장된 값을 사용
    classifierCNN.set_model(model_name="CNN", units=8, init="HeNormal", input_shape=[x_train.shape[1]],
                            activation=["relu", "relu", "sigmoid"], learning_rate=1e-4, optimizer="Adam",
                            loss=loss_func, epochs=30, batch_size=32)

    history.append(classifierCNN.start_learning(x_train, x_test, y_train, y_test))

    y_pred = classifierCNN.predict(x_test)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    cnn_acc.append(metrics.accuracy_score(y_test, y_pred_binary))
    cnn_precision.append(metrics.precision_score(y_test, y_pred_binary))
    cnn_recall.append(metrics.recall_score(y_test, y_pred_binary))
    cnn_f1_score.append(metrics.f1_score(y_test, y_pred_binary))


# Set up the data for plotting
data = {
    'Accuracy': {
        'SimpleRNN': {
            'binary_crossentropy': rnn_acc[0],
            'categorical_crossentropy': rnn_acc[1],
            'mean_squared_error': rnn_acc[2]
        },
        'CNN': {
            'binary_crossentropy': cnn_acc[0],
            'categorical_crossentropy': cnn_acc[1],
            'mean_squared_error': cnn_acc[2]
        }
    },
    'Precision': {
        'SimpleRNN': {
            'binary_crossentropy': rnn_precision[0],
            'categorical_crossentropy': rnn_precision[1],
            'mean_squared_error': rnn_precision[2]
        },
        'CNN': {
            'binary_crossentropy': cnn_precision[0],
            'categorical_crossentropy': cnn_precision[1],
            'mean_squared_error': cnn_precision[2]
        }
    },
    'Recall': {
        'SimpleRNN': {
            'binary_crossentropy': rnn_recall[0],
            'categorical_crossentropy': rnn_recall[1],
            'mean_squared_error': rnn_recall[2]
        },
        'CNN': {
            'binary_crossentropy': cnn_recall[0],
            'categorical_crossentropy': cnn_recall[1],
            'mean_squared_error': cnn_recall[2]
        }
    },
    'F1 Score': {
        'SimpleRNN': {
            'binary_crossentropy': rnn_f1_score[0],
            'categorical_crossentropy': rnn_f1_score[1],
            'mean_squared_error': rnn_f1_score[2]
        },
        'CNN': {
            'binary_crossentropy': cnn_f1_score[0],
            'categorical_crossentropy': cnn_f1_score[1],
            'mean_squared_error': cnn_f1_score[2]
        }
    }
}

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['#FF7F50', '#1E90FF'] 
# Plot the data
for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    metric_data = data[metric]
    for j, model_name in enumerate(models):
        axs[row, col].bar(model_name, metric_data[model_name][loss_functions[j]], label=loss_functions[j], color=colors[j])
    axs[row, col].set_title(metric)
    axs[row, col].set_ylim([0, 1])
    axs[row, col].legend(loc='lower right')

plt.tight_layout()
plt.show()