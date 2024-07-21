import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_dataset_path = 'E:/Prodigy ML/Task03/train'

img_width, img_height = 64,64

def load_data(directory):
    images =[]
    labels =[]
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory,img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        images.append(img)
        if 'dog' in img_name:
            labels.append(0)
        elif 'cat' in img_name:
            labels.append(1)
    return np.array(images), np.array(labels)


#loading the data

images,labels = load_data(train_dataset_path)

#converting the images to grayscale for better output

images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
images_flat = [img.flatten() for img in images]

images_flat = np.array(images_flat)
labels = np.array(labels)


#splitting the data
X_train, X_test, y_train, y_test = train_test_split(images_flat, labels, test_size=0.2, random_state=42)

#training the model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

#predicting the data
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

#saving model
joblib.dump('svm_dogs_vs_cats.pkl')
