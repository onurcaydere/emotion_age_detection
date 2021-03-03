import sys, os
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model

emotion_dict = {0: "Kızgın", 1: "Nefret", 2: "Korkmuş", 3: "Mutlu", 4: "Üzgün", 5: "Şaşırmış", 6: "Doğal"}

model = load_model("C:/Users/xxx/Desktop/Bitirme1/Dataface_model1.h5")


test_img_path ="C:/Users/xxx/Desktop/Bitirme1/test.JPG"

frame = cv2.imread(test_img_path)

gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


face_cascade = cv2.CascadeClassifier('D:/Users/xxx/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(cropped_img)

cv2.waitKey(0)
cv2.destroyAllWindows() 
