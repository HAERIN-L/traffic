import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

## flag, protocol, state
# 데이터 로드 및 전처리
df_1_1 = pd.read_csv("IoT_Botnet_Dataset_50.csv")
df_7_3 = pd.read_csv("IoT_Botnet_Dataset_70.csv")
df_9_1 = pd.read_csv("IoT_Botnet_Dataset_90.csv")

# 필요한 피처 가져오기
X_df_1_1 = df_1_1[['attack', 'spkts', 'pkts', 'sbytes', 'dpkts', 'bytes', 'dbytes', 'drate', 'drate', 'srate', 'rate', 'dur', 'stime', 'ltime', 'sum']]
X_1_1 = X_df_1_1.drop(['attack'], axis=1) # 학습
y_1_1 = X_df_1_1['attack'] # 결과

X_df_7_3 = df_7_3[['attack', 'spkts', 'pkts', 'sbytes', 'dpkts', 'bytes', 'dbytes', 'drate', 'drate', 'srate', 'rate', 'dur', 'stime', 'ltime', 'sum']]
X_7_3 = X_df_7_3.drop(['attack'], axis=1) # 학습
y_7_3 = X_df_7_3['attack'] # 결과

X_df_9_1 = df_9_1[['attack', 'spkts', 'pkts', 'sbytes', 'dpkts', 'bytes', 'dbytes', 'drate', 'drate', 'srate', 'rate', 'dur', 'stime', 'ltime', 'sum']]
X_9_1 = X_df_9_1.drop(['attack'], axis=1) # 학습
y_9_1 = X_df_9_1['attack'] # 결과

# 데이터셋을 훈련 데이터와 검증 데이터로 분할
# 2:3
X_train_1_1, X_val_1_1, y_train_1_1, y_val_1_1 = train_test_split(X_1_1, y_1_1, test_size=0.4, random_state=42)
X_train_7_3, X_val_7_3, y_train_7_3, y_val_7_3 = train_test_split(X_7_3, y_7_3, test_size=0.4, random_state=42)
X_train_9_1, X_val_9_1, y_train_9_1, y_val_9_1 = train_test_split(X_9_1, y_9_1, test_size=0.4, random_state=42)

# 피처 스케일링
scaler = StandardScaler()
X_train_scaled_1_1 = scaler.fit_transform(X_train_1_1)
X_val_scaled_1_1 = scaler.transform(X_val_1_1)

X_train_scaled_7_3 = scaler.fit_transform(X_train_7_3)
X_val_scaled_7_3 = scaler.transform(X_val_7_3)

X_train_scaled_9_1 = scaler.fit_transform(X_train_9_1)
X_val_scaled_9_1 = scaler.transform(X_val_9_1)


#모델링
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(128, input_dim=X_train_1_1.shape[1], activation='relu'),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

#컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#모델 학습
history_1_1 = model.fit(X_train_scaled_1_1, y_train_1_1, epochs=20, batch_size=64, validation_data=(X_val_scaled_1_1, y_val_1_1))
history_7_3 = model.fit(X_train_scaled_7_3, y_train_7_3, epochs=20, batch_size=64, validation_data=(X_val_scaled_7_3, y_val_7_3))
history_9_1 = model.fit(X_train_scaled_9_1, y_train_9_1, epochs=20, batch_size=64, validation_data=(X_val_scaled_9_1, y_val_9_1))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Loss 그래프 그리기
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_ylim(0, 1.5)
axes[0].grid()
for i in range(3):
    axes[0].plot(history[i].history['loss'], label=f'train({i})')
    axes[0].plot(history[i].history['val_loss'], label=f'val({i})')
axes[0].legend(loc='upper right')

# Accuracy 그래프 그리기
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
axes[1].grid()
for i in range(3):
    axes[1].plot(history[i].history['accuracy'], label=f'train({i})')
    axes[1].plot(history[i].history['val_accuracy'], label=f'val({i})')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()