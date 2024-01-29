import sys

from GUI import *
import tensorflow as tf

import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Dostępne urządzenia GPU:")
print(tf.config.list_physical_devices('GPU'))
print("Wersja TensorFlow-GPU:", tf.__version__)

model = load_model('football_events_model_10_2024-01-05_10_17_12.h5')
# #Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig(f"confusion_matrix_resnet50_final.png")

img_height, img_width = 224, 224
batch_size = 32

def prepare_train_data(train_img_dir :str = 'Train/train_catalogs'):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        train_img_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator

def prepare_test_data(test_img_dir : str = 'Test/test_catalogs'):
    generate_validation_data = ImageDataGenerator(rescale=1./255)
    validation_generator = generate_validation_data.flow_from_directory(
        test_img_dir,
        target_size=(img_height, img_width), 
        batch_size=batch_size,
        class_mode='categorical')
    
    return validation_generator

train_generator = prepare_train_data()
testing_generator = prepare_test_data()

#Print the Target names
from sklearn.metrics import classification_report, confusion_matrix
import itertools 
#shuffle=False
target_names = []
for key in train_generator.class_indices:
    target_names.append(key)

print(target_names)
# print(target_names)
#Confution Matrix
Y_pred = model.predict(testing_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(testing_generator.classes, y_pred)


# print(cm)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
# Print Classification Report
print('Classification Report')
print(classification_report(testing_generator.classes, y_pred, target_names=target_names))

def main() -> int:
    return 0

if __name__ == '__main__':
    sys.exit(main())


