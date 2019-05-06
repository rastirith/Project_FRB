#TensorFlow CNN testing
#Varying models of CNN with training data
#Assessing perfomance with TensorBoard

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import time
import pickle

#loading in training data

pickle_in = open("generated_data_7\\X_scaled.pickle","rb")
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("generated_data_7\\y2.pickle","rb")

y = pickle.load(pickle_in)
pickle_in.close()

#TensorFlow part of code

#model variations
dense_layers = [1]

dense_layer_sizes = [128]
conv_layer_sizes = [64]

conv_layers = [3]

print(len(X))
print(X.shape[1:])

#looping over all model variations
for dense_layer_size in dense_layer_sizes:
    for dense_layer in dense_layers:
        for conv_layer_size in conv_layer_sizes:
            for conv_layer in conv_layers:

                Name = f"binary_dropout_40_{conv_layer}_conv_{conv_layer_size}_nodes_{dense_layer}_dense_{dense_layer_size}_nodes_{int(time.time())}"

                print(Name)
                #TensorBoard callback
                
                tensorboard = TensorBoard(log_dir=f"logs_100/{Name}")
                
                earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

                model = Sequential()
                
                model.add(Conv2D(conv_layer_size, (3,3), input_shape = X.shape[1:]))
                model.add(Activation("relu"))

                model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Dropout(0.4))

                
                for l in range(conv_layer-1):
                    model.add(Conv2D(conv_layer_size, (3,3)))
                    model.add(Activation("relu"))

                    model.add(MaxPooling2D(pool_size=(2,2)))

                    model.add(Dropout(0.4))

                    
                model.add(Flatten())
                for l in range(dense_layer):
                    model.add(Dense(dense_layer_size))
                    model.add(Activation("relu"))

                    model.add(Dropout(0.4))

    
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
                
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                

                model.fit(X, y, epochs=20, batch_size = 128, callbacks=[tensorboard, earlystop], validation_split=0.15) ##validate here for tensorboard callback or use callback param in evaluate

               
    
    
    





