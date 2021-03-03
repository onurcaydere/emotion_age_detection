import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import Sequential,Model,model_from_json
from keras.layers import Dense,Conv2D, Activation, MaxPool2D, Flatten,Dropout,BatchNormalization
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint 
 
root="C:\\Users\\xxx\\Desktop\\Bitirme1\\Data\\fer2013.csv"
data=pd.read_csv(root)
data.shape



np.unique(data["Usage"].values.ravel())
print("Eğitim setindeki örnek sayısı %d"%(len(data[data.Usage=="Training"])))

train_data=data[data.Usage=="Training"]

#Eğitim Örneklerinin piksel değerleri bize tablo halinde yan yana verildiği için boşluklardan parse ederek liste haline dönüştürdük

train_pixel=train_data.pixels.str.split(" ").tolist()
train_pixel=pd.DataFrame(train_pixel,dtype=int)
train_images=train_pixel.values
train_images=train_images.astype(np.float)


#Görüntüyü 48*48 piksel şeklinde göstermek için fonk yazıyoruz

def show(img):
    show_image=img.reshape(48,48)
    
    plt.axis("off")
    plt.imshow(show_image,cmap='gray')

train_labels_flat = train_data["emotion"].values.ravel()
train_labels_count = np.unique(train_labels_flat).shape[0]
print('Farklı yüz ifadelerinin sayısı: %d'%train_labels_count)


np.unique(data["Usage"].values.ravel()) 
test_data = data[data.Usage == "PublicTest"] 
test_pixels = test_data.pixels.str.split(" ").tolist() 

test_pixels = pd.DataFrame(test_pixels, dtype=int)
test_images = test_pixels.values
test_images = test_images.astype(np.float)

print(test_images.shape)

test_labels_flat = test_data["emotion"].values.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]
num_classes=7
def dense_to_one_hot(labels_dense,num_classed):
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot

y_test = dense_to_one_hot(test_labels_flat, test_labels_count)
y_test = y_test.astype(np.uint8)
print(y_test.shape)




model = Sequential()
#1. Katman
#channel last önce yükseklik sonra genişlik sonrada kanal bilgisi olacak 
#başlangıç kernelleri he_normal
#input shape 48,48,1 

model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
# 2 . katman 
#maxpooling (max ortaklama işlemi kullandık ) pool size kaç pixel aralığını pooling yapacak bunu belirledik
#adım aralığı yaptık
#nöron silme işlemini yaptık 

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))
#3. katmanda 
#Kanl sayısını azaltabiliriz
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
#aynısını buradada yaptım
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
#maxpooling ve dropout işlemlerini yaptık yine
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))
#tam bağlantı katmanı vektorizasyon işlemini yapıyoruz.
#flatten düzleştirme 
#Dense ===Sinir ağı uyguladık 2048 lik flatten vektörünü 128 eindirdik

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))
#Drop outtan sonra delen sinir ağını 7 ye düşürdük  sebebi 7 sınıfım var.
#•Dropout genellikle 0,25 ve 0,5 arasında kullanılır 
#Sınıflandırıcı yapabileceğimiz bir activasyon seçmem lazımdı bu yüzden softmax kullanlıyorum .

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=["accuracy"])
# Modelimizin özetini göreceğiz 
model.summary()
#Eğitim ve test kümelerindeki eleman sayısı yükseklik genişlik ve kanal sayısını yazdırdık
x_train=train_images.reshape(-1,48,48,1)
x_test=test_images.reshape(-1,48,48,1)
print("Train:",x_train.shape)
print("Test:",x_test.shape)
#Model Eğitimi en başarılı olanları kaydetmem lazım :
kaydetme_yolu="C:\\Users\\xxx\\Desktop\\Bitirme1\\Data"

checkpointer=ModelCheckpoint(filepath=kaydetme_yolu+"face_model.h5",verbose=1,save_best_only=True)
epochs=50   
batchSize=100

hist=model.fit(x_train,y_train,epochs=epochs,shuffle=True,batch_size=batchSize,
               validation_data=(x_test,y_test),callbacks=[checkpointer], verbose=2)
#Modelimizi json olarak kaydedelim çünkü daha sonradan eğitim yapmadan kullanabileleim

model_json=model.to_json()
with open(kaydetme_yolu+"modelim.json",'w') as json_file:
    json_file.write(model_json)

plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
plt.suptitle("Train")
plt.ylabel("Loss")
plt.plot(hist.history['loss'],color="r",label="Training Loss")
plt.plot(hist.history['val_loss'],color="b",label="Validation Loss")
plt.legend(loc="upper right")




plt.subplot(1,2,2)
plt.suptitle("Accuracy")
plt.ylabel("Loss")
plt.plot(hist.history['acc'],color="g",label="Training Accuracy")
plt.plot(hist.history['val_acc'],color="m",label="Validation Accuracy")
plt.legend(loc="lower right")

plt.show()


