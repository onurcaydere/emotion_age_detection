import sys, os
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
import dlib

emotion_dict = {0: "Kizgin", 1: "İgrenmis", 2: "Korkmus", 3: "Mutlu", 4: "Uzgun", 5: "Sasirmis", 6: "Poker Face"}
model = load_model("egitim_modellerim\\gender_model1.h5")
model2=load_model("egitim_modellerim\\Dataface_model1.h5")

detector = dlib.get_frontal_face_detector()
cap= cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3



while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray)

    for det in dets:
        cv2.rectangle(frame,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
        roi_gray = gray[det.top():det.bottom(), det.left():det.right()]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model2.predict(cropped_img)
        prediction2=model.predict(cropped_img)
        
        if(prediction2<0.5):
            cv2.putText(frame,"Erkek",(det.right(),det.top()),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame,"Kadin",(det.right(),det.top()),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (det.left()+25, det.bottom()+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Duygu Ve Analizi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
