from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import losses
from keras.applications import VGG16

from keras.preprocessing import image
import glob
import numpy as np
from keras.utils import np_utils


def load_GTSRB():
    imlist = []
    GT = [] #ground truth
    
    C1files = glob.glob(r'''C:\ml\GTSRB_subset_2\class1\*.jpg''' )
    C2files = glob.glob(r'''C:\ml\GTSRB_subset_2\class2\*.jpg''' )
    
    for file in C1files:
        imlist.append(np.array(image.load_img(file)))
        GT.append(0)
    
    for file in C2files:
        imlist.append(np.array(image.load_img(file)))
        GT.append(1)
    
    return np.asarray(imlist),GT

X,y = load_GTSRB()
y = np_utils.to_categorical(y,2)


base_model = VGG16(include_top=False, weights = "imagenet",
                   input_shape = (64,64,3))

w = base_model.output

w = Flatten()(w)

w = Dense(100,activation= "relu")(w)

output = Dense(2, activation = "sigmoid")(w)

model = Model(inputs = [base_model.input], outputs = [output])

model.layers[-5].trainable = True
model.layers[-6].trainable = True
model.layers[-7].trainable = True


model.summary()
model.compile(optimizer = "sgd",metrics=['accuracy'], loss = 'binary_crossentropy')

model.fit(X, y, epochs=2, batch_size=32)
