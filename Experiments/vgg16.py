# import libs

from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D as MaxPool2D, Dense, Flatten, Dropout, Resizing, Rescaling, RandomFlip, RandomRotation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# open sataset 
DIMENSION = 350
BATCH = 1

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="/dataset_flowers/train",batch_size = BATCH,target_size=(DIMENSION,DIMENSION))

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="/dataset_flowers/test", batch_size = BATCH, target_size=(DIMENSION,DIMENSION))


# load VGG16 pre trained model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(DIMENSION,DIMENSION,3))
base_model.trainable = False


# adapt the model to my problem
flatten_layer = Flatten()

dropout = Dropout(0.2)
resize_and_rescale = Sequential([
  Resizing(DIMENSION, DIMENSION),
  Rescaling(1./255)
])

data_augmentation = Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),
])


dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(6, activation='softmax')


model = Sequential([
    resize_and_rescale,
    data_augmentation,
    base_model,
    dropout,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

opt = Adam(lr=0.001)


# run model
model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])

model.fit_generator(
    generator=traindata, 
    validation_data=testdata, 
    steps_per_epoch=int(48/BATCH), 
    validation_steps=int(12/BATCH), 
    epochs=30
)

'''
# apply model to classification task
def load_image(filename):
    img = load_img(filename, target_size=(DIMENSION, DIMENSION))
    img = img_to_array(img)
    img = img.reshape(1, DIMENSION, DIMENSION, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

img = load_image('Jabuticaba.7.jpg')
model2.predict(img)[0]

'''
  