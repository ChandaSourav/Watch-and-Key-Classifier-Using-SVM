import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import cv2 as cv

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def trainModel(self):
        img_list = np.array([])
        class_list = np.array([])

        for i in range(1, 51):
                    img = cv.imread(f'key/frame{i}.jpg')[:,:,0]
                    img = img.reshape(150, 113) 
                    img_list = np.append(img_list, [img])
                    class_list = np.append(class_list, 1)
        for j in range(1, 51):
                    img = cv.imread(f'watch/frame{j}.jpg')[:,:,0]
                    img = img.reshape(150, 113) 
                    img_list = np.append(img_list, [img])
                    class_list = np.append(class_list, 2)


        img_list = img_list.reshape(100, 16950)

        self.model = LinearSVC(random_state=1)

        self.model.fit(img_list,class_list)

        print("Model trained successfully.")


    def predict(self, frame):
        img = cv.imread(frame)[:, :, 0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])

        return prediction

model = Model()

model.trainModel()
prediction = model.predict('watch/frame1.jpg')
prediction = int(prediction[0])
classes=['Key', 'Watch']

print(classes[prediction-1])








