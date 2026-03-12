import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.data.experimental import unique

X_train = np.load('./binary_X_train.npy', allow_pickle=True)
X_test = np.load('./binary_X_test.npy', allow_pickle=True)
Y_train = np.load('./binary_Y_train.npy', allow_pickle=True)
Y_test = np.load('./binary_Y_test.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Conv2D(32, input_shape=(128, 128, 3), kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.summary()

# 1. 데이터의 생김새와 처음 10개 값 확인
print("Y_train 모양(Shape):", Y_train.shape)
print("Y_train 샘플 10개:", Y_train[:10])

# 2. 어떤 종류(클래스)가 있는지, 각 몇 개씩 있는지 확인 (가장 중요!)
unique_classes, counts = np.unique(Y_train, return_counts=True)
print("클래스 종류와 개수:", dict(zip(unique_classes, counts)))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7) # 검증 accuracy가 7epoch 동안 좋아지지 않다면 학습을 멈춤
fit_hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.15, callbacks=[early_stopping])

score = model.evaluate(X_test,Y_test)
print('Evaluation loss: ', score[0])
print('Evaluation accuracy : ', score[1])

model.save('./spec_rotor_{}.h5'.format(score[1]))
plt.plot(fit_hist.history['loss'], label = 'loss')
plt.plot(fit_hist.history['val_loss'], label = 'validation loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'], label = 'train accuracy')
plt.plot(fit_hist.history['val_accuracy'], label = 'validation accuracy')
plt.legend()
plt.show()

# 1. 모델 예측 (확률값 -> 정수 클래스로 변환)
Y_pred_prob = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred_prob, axis=1) # [0.1, 0.9...] -> 1

# 2. 카테고리 이름 (위에서 정의했던 것 그대로 사용)
categories = ['Healthy', 'Fault10', 'Fault30', 'Fault60']

# 3. 혼동 행렬 생성
cm = confusion_matrix(Y_test, Y_pred_classes)

# 4. 시각화 (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)

plt.title('Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 1. 틀린 인덱스 찾기
diff_indices = np.where(Y_test != Y_pred_classes)[0]

# 2. 틀린 것 중 5개만 뽑아서 그림 그리기
plt.figure(figsize=(15, 5))
num_view = 5

for i in range(min(len(diff_indices), num_view)):  # 틀린 게 5개 미만일 수도 있으니 min 사용
    idx = diff_indices[i]

    plt.subplot(1, num_view, i + 1)
    plt.imshow(X_test[idx])  # 원본 이미지
    plt.axis('off')

    # 제목에 정답과 예측값 표시
    true_name = categories[Y_test[idx]]
    pred_name = categories[Y_pred_classes[idx]]

    plt.title(f"True: {true_name}\nPred: {pred_name}", color='red')

plt.suptitle(f'Wrong Predictions (Total: {len(diff_indices)})', fontsize=16)
plt.tight_layout()
plt.show()

