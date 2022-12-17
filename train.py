# import the libraries
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set the working directory (Make sure you have the data in your working directory under the same name)
train_dir = './chest_xray/train'
val_dir = './chest_xray/val'

# rescale the data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# get the data from the training directory
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    shuffle=True)
# get the validation data from the validation directory
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    shuffle=True)

# create a function for training the model
def make_model(learning_rate=0.001, size_inner=64, droprate=0.0):
    
    ####################################################
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(droprate))
    model.add(layers.Dense(size_inner, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    ###################################################
    
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.8)
    loss='binary_crossentropy'
    model.compile(loss=loss,
                 optimizer=optimizer,
                 metrics=['acc'])
    return model

# create the checkpoint to store the data with the highest accuracy
checkpoint = keras.callbacks.ModelCheckpoint(
    'pneumonia-class_v1_{epoch:02d}_{val_acc:.3f}.h5',
    save_best_only=True,
    monitor='val_acc',
    mode='max',
    
)

# use the hyperparameters that achieve the highest performance
learning_rate = 0.001
droprate = 0.8
size = 100

model = make_model(learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )
    
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

model = keras.models.load_model('pneumonia-class_v1_03_0.938.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('pneumonia-class.tflite', 'wb') as f_out:
    f_out.write(tflite_model)