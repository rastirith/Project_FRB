#Matrix data parsing attempt into CNN

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import time
import os
import random
import pickle

DATADIR = os.getcwd() + "\matrix_files\\Playground2\\"
training_data=[]

#CATEGORIES=["Signal","No_signal"]
CATEGORIES=["Noise","Burst"]

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for matrix in os.listdir(path):
            if len(training_data) == 40:
                break
            try:
                matrix_arr = np.load(os.path.join(path,matrix))
                training_data.append([matrix_arr, class_num])
            except:
                pass
       
create_training_data()      

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 70, 100, 1)

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


#TensorFlow part of code
Name = "Matrix-CNN-64x2-256Dense-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(Name))

print(X.shape[1:])

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation("relu"))

model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, epochs=1,callbacks=[tensorboard], validation_split=0.3) ##validate here for tensorboard callback or use callback param in evaluate




