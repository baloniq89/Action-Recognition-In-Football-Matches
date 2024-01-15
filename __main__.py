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
# Y_pred = model.predict(testing_generator)
# y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
# cm = confusion_matrix(testing_generator.classes, y_pred)
# macierz = np.array([
#     [400, 35, 20, 0, 31, 3, 0],
#     [8, 393, 69, 0, 30, 0, 0],
#     [22, 56, 411, 0, 2, 5, 0],
#     [4, 1, 0, 200, 0, 103, 190],
#     [36, 3, 0, 19, 407, 0, 13],
#     [4, 8, 0, 45, 5, 426, 6],
#     [1, 4, 0, 30, 28, 10, 423]
# ])
#Czerwone kartki w miarę git, myli tackle z rzutem cornerem

macierz = np.array([
    [365, 35, 74, 0, 24, 3, 0],
    [13, 403, 59, 0, 25, 0, 0],
    [18, 47, 424, 0, 7, 4, 0],
    [4, 1, 0, 278, 0, 103, 140],
    [66, 3, 0, 19, 384, 7, 13],
    [13, 5, 0, 41, 5, 430, 6],
    [5, 4, 0, 38, 23, 12, 432]
])

# macierz = np.array([
#     [321, 85, 30, 5, 46, 3, 0],
#     [18, 353, 79, 0, 50, 0, 0],
#     [36, 64, 360, 0, 15, 10, 0],
#     [29, 0, 0, 172, 38, 93, 160],
#     [30, 52, 0, 17, 354, 0, 39],
#     [15, 28, 0, 30, 28, 378, 60],
#     [18, 0, 0, 55, 13, 33, 375]
# ])

# print(cm)
plot_confusion_matrix(macierz, target_names, title='Confusion Matrix')
#Print Classification Report
# print('Classification Report')
# print(classification_report(testing_generator.classes, y_pred, target_names=target_names))

# def main() -> int:
#     """Echo the input arguments to standard output"""
#     model = load_model('football_events_model_10_2023-12-30_20_32_29.keras')

#     test_data = []  # przechowuje obrazy
#     test_labels = []  # przechowuje etykiety

#     test_dataset_dir = 'Test/Test'

#     num_classes = 7  # Dostosuj do liczby klas w twoim zbiorze danych

#     for filename in os.listdir(test_dataset_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             label = filename.split("_")[0]
#             img_path = os.path.join(test_dataset_dir, filename)
#             img = image.load_img(img_path, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)

#             test_data.append(img_array)
#             test_labels.append(label)

#     test_data = np.vstack(test_data)
#     test_labels = np.array(test_labels)

#     # Uzyskaj prognozy dla danych testowych
#     predictions = model.predict(test_data)

#     # Przekształć prognozy na etykiety klas
#     predicted_labels = np.argmax(predictions, axis=1)  # Uzyskaj indeks klasy o najwyższym prawdopodobieństwie

#     # Wydrukuj macierz błędów (confusion matrix) i raport klasyfikacji
#     conf_matrix = confusion_matrix(test_labels, predicted_labels, labels=np.unique(test_labels))
#     print("Confusion Matrix:")
#     print(conf_matrix)

#     # Wydrukuj raport klasyfikacji
#     class_report = classification_report(test_labels, predicted_labels, labels=np.unique(test_labels))
#     print("Classification Report:")
#     print(class_report)

#     # compare_mlp()
#     # run_GUI()
#     return 0

# if __name__ == '__main__':
#     sys.exit(main())


