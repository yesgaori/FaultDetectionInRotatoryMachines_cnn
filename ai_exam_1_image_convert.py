# 이미지 전처리
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

img_dir = '/media/user13/data/archive/'
categories = ['spec_healthy', 'spec_rotor_fault10', 'spec_rotor_fault30', 'spec_rotor_fault60']

image_w = 128
image_h = 128

pixel = image_h * image_w * 3 # RGB channel

X = []
Y = []
files = None

for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '*.png')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            data = img.resize((image_w, image_h))
            X.append(data)
            Y.append(idx)
            if i % 300 == 0: # 파일이 많아서, 다 끝날때까지 잘 되고있는지 확인하는 용도.
                print(category, ':', f)
        except:
            print(category, i, 'error')

X = np.array(X)
Y = np.array(Y)
X = X / 255 # 스케일링
print(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
np.save('./binary_X_train.npy', X_train)
np.save('./binary_X_test.npy', X_test)
np.save('./binary_Y_train.npy', Y_train)
np.save('./binary_Y_test.npy', Y_test)

# 한글 폰트 설정 (필요시 운영체제에 맞게 설정, 영문으로 할 거면 생략 가능)
# plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(10, 6))
unique, counts = np.unique(Y, return_counts=True)
sns.barplot(x=categories, y=counts, palette='viridis')

plt.title('Number of Samples per Category', fontsize=15)
plt.ylabel('Count')
plt.xlabel('Category')
# 막대 위에 숫자 표시
for i, v in enumerate(counts):
    plt.text(i, v + 5, str(v), ha='center', fontsize=12)

plt.show()

plt.figure(figsize=(15, 4))

for i, category in enumerate(categories):
    # 해당 카테고리의 첫 번째 이미지 인덱스 찾기
    idx = np.where(Y == i)[0][0]

    plt.subplot(1, 4, i + 1)
    plt.imshow(X[idx])  # 이미 스케일링(0~1) 되어있어도 imshow는 잘 보여줍니다.
    plt.title(category)
    plt.axis('off')  # 축 제거 (깔끔하게)

plt.suptitle('Spectrogram Sample for Each Class', fontsize=16)
plt.tight_layout()
plt.show()

sizes = [len(X_train), len(X_test)]
labels = ['Train Set', 'Test Set']
colors = ['#ff9999', '#66b3ff'] # 예쁜 파스텔 톤

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 14})
plt.title('Data Split Ratio', fontsize=16)
plt.show()