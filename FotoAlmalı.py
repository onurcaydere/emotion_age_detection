import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator




objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')

model = load_model("C:/Users/xxx/Desktop/Bitirme1/Dataface_model1.h5")

img_path="C:/Users/xxx/Desktop/Bitirme1/kemalsunal.jpg"

test_img=image.load_img(img_path)

img= image.load_img(img_path,grayscale=True,target_size=(48,48))

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x /= 255

custom=model.predict(x)

y_pos=np.arange(len(objects))

plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')
plt.show()

#2
x = np.array(x, 'float32')
x = x.reshape([48, 48]);
plt.axis('off')
plt.gray()
plt.imshow(test_img)

plt.show()

