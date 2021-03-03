import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Conv2D, Activation, MaxPooling2D, Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split

#Veri setimi Porjeye dahil ediyorum.
data = pd.read_csv('C:\\Users\\xxx\\Desktop\\Bitirme1\\Data\\age_gender.csv')

## Converting pixels into numpy array
data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

data.head()

print('Total Satır: {}'.format(len(data)))
print('Total Sütun: {}'.format(len(data.columns)))


X = np.array(data['pixels'].tolist())

## 1D lik pixeli 3D ye dönüştürüyoruz.
X = X.reshape(X.shape[0],48,48,1)


#Gender için Eğitim Modeli
y = data['gender']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=37
)
model=Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0,5))
model.add(Dense(1,activation="sigmoid"))




model.compile(loss='binary_crossentropy',optimizer="sgd",metrics=["accuracy"])


model.summary()

model_kayit_path="C:\\Users\\xxx\\Desktop\\Bitirme1\\"

checkpointer=ModelCheckpoint(filepath=model_kayit_path+"gender_model1.h5",verbose=1,save_best_only=True)

history = model.fit(
    X_train, y_train, epochs=30, validation_split=0.1, batch_size=256, callbacks=[checkpointer]
)

plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
plt.suptitle("Train")
plt.ylabel("Loss")
plt.plot(history.history['loss'],color="r",label="Training Loss")
plt.plot(history.history['val_loss'],color="b",label="Validation Loss")
plt.legend(loc="upper right")


plt.subplot(1,2,2)
plt.suptitle("Accuracy")
plt.ylabel("Loss")
plt.plot(history.history['acc'],color="g",label="Training Accuracy")
plt.plot(history.history['val_acc'],color="m",label="Validation Accuracy")
plt.legend(loc="lower right")


loss, acc = model.evaluate(X_test,y_test,verbose=0)
print('Test Kayıp: {}'.format(loss))
print('Test Doğruluk: {}'.format(acc))
####Gender Eğitimi Başarılı Sonuç Verdi.






