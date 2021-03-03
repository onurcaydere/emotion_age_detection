import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import dlib

emotion_dict = {0: "Kizgin", 1: "Nefret", 2: "Korkmus", 3: "Mutlu", 4: "Uzgun", 5: "Sasirmis", 6: "Poker Face"}
mapper=["Erkek","Kadin"]
model = load_model("egitim_modellerim\\gender_model1.h5")
model2=load_model("egitim_modellerim\\Dataface_model1.h5")
detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3

img = cv2.imread('image\\kemalsunal.JPG',0)
plt.imshow(img, cmap="gray")
det = detector(img)
cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)


img = cv2.resize(img, (48,48))
img = np.reshape(img,[1,48,48,1])
img_pixels = img.astype("float32") / 255.0
classes = model.predict_classes(img_pixels)
prediction=model.predict(img)
mapper=['Erkek','Kadın']
if(prediction>=0.5):
    print(mapper[1])
else:
    print(mapper[0])