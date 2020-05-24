from keras.datasets import mnist
from keras.utils import np_utils
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import yaml
import struct
import numpy as np
def read_idx(filename):
        """Credit: https://gist.github.com/tylerneylon"""
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def vishyaml1(i):
    with open('vish1.yaml') as f:
    
        vishdocs = yaml.load_all(f, Loader=yaml.FullLoader)
    
        for doc in vishdocs:
            
            for key, value in doc.items():
                vishvalu = value[0:i]	
    return vishvalu
def vishyaml2(i):
    with open('vish2.yaml') as f:
    
        vishdocs = yaml.load_all(f, Loader=yaml.FullLoader)
    
        for doc in vishdocs:
            
            for key, value in doc.items():
                vishvalu = value[0:i]	
    return vishvalu

def vishmodeltrain(num_classes,input_shape,i):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(BatchNormalization())
    value=vishyaml2(i)
    for j in value:
        exec(j)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    
    value=vishyaml1(i)
    for j in value:
        exec(j)
    
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
    print(model.summary())
    return model


def vishloaddata():
    x_train = read_idx("./fashion_mnist/train-images-idx3-ubyte")
    y_train = read_idx("./fashion_mnist/train-labels-idx1-ubyte")
    x_test = read_idx("./fashion_mnist/t10k-images-idx3-ubyte")
    y_test = read_idx("./fashion_mnist/t10k-labels-idx1-ubyte")
    
    print("Initial shape or dimensions of x_train", str(x_train.shape))
    print ("Number of samples in our training data: " + str(len(x_train)))
    print ("Number of labels in our training data: " + str(len(y_train)))
    print ("Number of samples in our test data: " + str(len(x_test)))
    print ("Number of labels in our test data: " + str(len(y_test)))
    print()
    print ("Dimensions of x_train:" + str(x_train[0].shape))
    print ("Labels in x_train:" + str(y_train.shape))
    print()
    print ("Dimensions of x_test:" + str(x_test[0].shape))
    print ("Labels in y_test:" + str(y_test.shape))
    
    # Training Parameters
    batch_size = 128
    epochs = 3

    # Lets store the number of rows and columns
    img_rows = x_train[0].shape[0]
    img_cols = x_train[1].shape[0]

    # Getting our date in the right 'shape' needed for Keras
    # We need to add a 4th dimenion to our date thereby changing our
    # Our original image shape of (60000,28,28) to (60000,28,28,1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # store the shape of a single image 
    input_shape = (img_rows, img_cols, 1)

    # change our image type to float32 data type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize our data by changing the range from (0 to 255) to (0 to 1)
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Now we one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Let's count the number columns in our hot encoded matrix 
    print ("Number of Classes: " + str(y_test.shape[1]))

    num_classes = y_test.shape[1]
    num_pixels = x_train.shape[1] * x_train.shape[2]

    i=0
    for i in range(10):
        model=vishmodeltrain(num_classes,input_shape,i)
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
       
        if score[1] < 0.95:
            continue
        else:
            break

vishloaddata()