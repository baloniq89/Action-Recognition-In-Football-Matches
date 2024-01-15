import os
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import tensorflow as tf
import PIL
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile, UnidentifiedImageError
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as pl
from datetime import date, datetime
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import plot_model
from visualkeras import layered_view
from tensorflow.keras.applications import VGG16
from PIL import ImageFont

import numpy as np
import visualkeras
import pandas as pd

# model = ResNet50(weights='imagenet', include_top=True)

# Wizualizacja za pomocą plot_model z TensorFlow
# plot_model(model, show_shapes=True, show_layer_names=True, to_file='resnet50.png')

# Wizualizacja za pomocą visualkeras
# visualkeras.layered_view(model, to_file='model_resnet50.png')

font = ImageFont.truetype("arial2.ttf", 30)

train_data_dir = 'Train/train_catalogs'
validation_data_dir = 'Test/test_catalogs'

# dataset_dir = 'Train/train_data'
dataset_dir = 'Train/Train'

img_height, img_width = 224, 224
epochs = 10
num_classes = 7
batch_size = 32

class_mapping = ["Corner", "FreeKick", "Penalty", "RedCards", "Tackle", "ToSubstitue", "YellowCards"]

def filter_images(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return image_files


def draw_curves(history, key1='accuracy', ylim1=(0.7, 1.00)):
    pl.figure(figsize=(12,4))
    pl.plot(history.history[key1], "r--")
    pl.plot(history.history['val_' + key1], "g--")
    pl.ylabel(key1)
    pl.xlabel('Epoch')
    pl.ylim(ylim1)
    pl.legend(['train', 'test'], loc='best')
    pl.savefig(f"model_accuracy_loss_plots/model_accuracy{time_stamp}_.png")

def draw_curves_loss(history, key1='loss', ylim1=(0.7, 1.00)):
    pl.figure(figsize=(12,4))
    pl.plot(history.history[key1], "r--")
    pl.plot(history.history['val_' + key1], "g--")
    pl.ylabel(key1)
    pl.xlabel('Epoch')
    pl.ylim(ylim1)
    pl.legend(['train', 'test'], loc='best')
    pl.savefig(f"model_accuracy_loss_plots/model_loss_{time_stamp}.png")

# create tested models 
def create_model_L1(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))) 
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

 

def create_model_L2(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, kernel_regularizer=regularizers.l2(0.01), activation='softmax'))
    return model

def create_model_dropout(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_model_simplify(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_model_mid(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_double_layer_model(img_height: int, img_width: int, num_classes: int):
    model = Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    return model

def VGG16(img_height: int, img_width: int, num_classes: int):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

#Use ImageDataGenerator to prepare data for train and test data sets

def generate_train_data():
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
    return train_datagen

def prepare_train_test_data(train_img_dir :str = 'Train/Train'):
    train_datagen = generate_train_data()
    train_generator = train_datagen.flow_from_directory(
        train_img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        train_img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )
    
    return train_generator, validation_generator



# ************** Data generators used if we would have images splitted to few directories -> one class equals one direcctory   **************
def prepare_train_data(train_img_dir :str = train_data_dir):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator
    
def prepare_test_data(test_img_dir : str = validation_data_dir):
    generate_validation_data = ImageDataGenerator(rescale=1./255)
    validation_generator = generate_validation_data.flow_from_directory(
        test_img_dir,
        target_size=(img_height, img_width), 
        batch_size=batch_size,
        class_mode='categorical')
    
    return validation_generator
# ****************************************************************************************************************

# ************** Data generators used if we would have images splitted to few directories -> one class equals one direcctory   **************

# def prepare_train_data(train_img_dir :str = train_data_dir):
#     generate_train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
#     filtered_train_images = filter_images(train_data_dir)
#     train_data = {'data': [], 'labels': []}    
#     for image_file in filtered_train_images:
#         image_path = os.path.join(train_data_dir, image_file)
#         img_array = load_and_preprocess_image(image_path)
#         if img_array is not None:
#             train_data['data'].append(img_array)
#             label = image_file.split("_")[0]
#             train_data['labels'].append(label)

#     train_generator = generate_train_data.flow(np.array(train_data['data']), np.array(train_data['labels']), batch_size=batch_size, class_mode='categorical')
#     generate_train_data.flow()
#     return train_generator

# def prepare_test_data(test_img_dir : str = validation_data_dir):
#     generate_validation_data = ImageDataGenerator(rescale=1./255)
#     filtered_test_images = filter_images(validation_data_dir)
#     test_data = {'data': [], 'labels': []}    
#     for image_file in filtered_test_images:
#         image_path = os.path.join(validation_data_dir, image_file)
#         img_array = load_and_preprocess_image(image_path)
#         if img_array is not None:
#             test_data['data'].append(img_array)
#             label = image_file.split("_")[0]
#             test_data['labels'].append(label)

#     validation_generator = generate_validation_data.flow(np.array(test_data['data']), np.array(test_data['labels']), batch_size=batch_size, class_mode='categorical')

#     return validation_generator

# train = prepare_train_data()
# validation = prepare_test_data()

# ******************** Preparing train and validation data stored in one directory ********************

data = []
labels = []

for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = filename.split("_")[0]  # Załóż, że etykieta znajduje się przed znakiem "_"
        data.append(filename)
        labels.append(label)

df = pd.DataFrame({"filename": data, "label": labels})

# Podział na zbiór treningowy i walidacyjny
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)
print(train_df)
print(validation_df)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generator danych treningowych
train= train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=dataset_dir,  # Ścieżka do katalogu z danymi treningowymi
    x_col="filename",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Generator danych walidacyjnych
validation = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    directory=dataset_dir,
    x_col="filename",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
# ******************** ------------------------------------------------------- ********************

print("\n Inna metoda")
# train, test = prepare_train_test_data()
print(train.class_indices)
print(train.samples)
print(validation.samples)

# model_1 = create_model_L1(img_height=img_height, img_width=img_width, num_classes=num_classes)
# visualkeras.layered_view(model_1, to_file='model_structure.png')
# model_2 = create_model_mid(img_height=img_height, img_width=img_width, num_classes=num_classes)
# visualkeras.layered_view(model_2, to_file='model_mid_structure.png')
# model_simplify = create_model_simplify(img_height=img_height, img_width=img_width, num_classes=num_classes)
# visualkeras.layered_view(model_simplify, to_file='model_simplify_structure.png')


def train_model():
    # net_history = []
    model = create_model_L2(img_height=img_height, img_width=img_width, num_classes=num_classes)
    visualkeras.layered_view(model, legend=True, font=font, to_file='model_cnn_model_.png').show()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    EarlyStop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # model training
    net_history = model.fit(
        train,
        steps_per_epoch=train.samples // batch_size,
        epochs=epochs,
        validation_data=validation,
        validation_steps=validation.samples // batch_size,
        callbacks=[EarlyStop],
        )
    # print_loss_and_accuracy(model)
    
    return model, net_history

def train_model_resnet():
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    model = models.Sequential()
    model.add(resnet_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))
    visualkeras.layered_view(resnet_model, legend=True, font=font, to_file='model_resnet_.png').show()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    EarlyStop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # # model training
    # net_history = model.fit(
    #     train,
    #     steps_per_epoch=train.samples // batch_size,
    #     epochs=epochs,
    #     validation_data=validation,
    #     validation_steps=validation.samples // batch_size,
    #     callbacks=[EarlyStop],
    #     )
    return model, net_history

# modelvgg = VGG16(img_height=img_height, img_width=img_width, num_classes=num_classes)
# visualkeras.layered_view(modelvgg, to_file='model_vgg16.png')

def train_model_VGG16():
    # net_history = []
    model = VGG16(img_height=img_height, img_width=img_width, num_classes=num_classes)
    visualkeras.layered_view(model, legend=True, font=font, to_file='model_vgg_16.png').show()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    EarlyStop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # model training
    net_history = model.fit(
        train,
        steps_per_epoch=train.samples // batch_size,
        epochs=epochs,
        validation_data=validation,
        validation_steps=validation.samples // batch_size,
        callbacks=[EarlyStop],
        )
    # print_loss_and_accuracy(model)
    
    return model, net_history

today = date.today()
cur_time = datetime.now().strftime("%H_%M_%S")
time_stamp = str(f"{today}_{cur_time}")
model, net_history = train_model()
model.save(f"football_events_model_{epochs}_{time_stamp}.keras")
model.save(f"football_events_model_{epochs}_{time_stamp}.h5")

print(net_history.history['accuracy'])
# draw_curves(net_history, key1='accuracy')
# draw_curves_loss(net_history, key1='loss')

basic_color_train, basic_color_valid = 'cyan', 'red'
legend_acc, legend_loss = [], []
pl.figure(figsize=(16, 6))
        # wykres porownania model accuracy
pl.subplot(1,2,1)
pl.plot(net_history.history['accuracy'], linewidth=1, color=basic_color_train)
pl.plot(net_history.history['val_accuracy'], linewidth=1, color=basic_color_valid)

pl.subplot(1,2,2)
pl.plot(net_history.history['loss'], linewidth=1, color=basic_color_train)
pl.plot(net_history.history['val_loss'], linewidth=1, color=basic_color_valid)


    # tworzenie legendy do obu wykresow
pl.subplot(1,2,1)
legend_acc.append(f"training")
legend_acc.append(f"validation")

legend_loss.append(f"training")
legend_loss.append(f"validation")

pl.title(f"Model accuracy:")
pl.ylabel('accuracy')
pl.xlabel('epoch')
pl.legend(legend_acc, fontsize=10)

pl.subplot(1,2,2)
pl.title(f"Model loss:")
pl.ylabel('loss')
pl.xlabel('epoch')
pl.legend(legend_loss, fontsize=10)
# pl.show()

today = date.today()
cur_time = datetime.now().strftime("%H_%M_%S")
time_stamp = str(f"{today}_{cur_time}")

pl.savefig(f"model_accuracy_loss_plots/{time_stamp}_accuracy_loss.png")

