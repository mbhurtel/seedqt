# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(64, (3, 3), input_shape = (32,32,3),padding = 'same',activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

classifier.add(Convolution2D(128, (3, 3),padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

classifier.add(Convolution2D(256, (3, 3),padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), padding = 'same'))

# Flattening Step
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(units = 1024, activation = 'relu'))
#classifier.add(Dropout(0.5))
classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   fill_mode='nearest',
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(32, 32),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(32, 32),
                                                batch_size=32,
                                                class_mode='categorical')

s = training_set.class_indices
print(s)

classifier.fit_generator(training_set,
                         steps_per_epoch=2500//32,
                         validation_data=test_set,
                         epochs=30,
                         validation_steps=500//32)
                         
classpath = "model.hdf5"
classifier.save(classpath)                                                                                                                                                                                     