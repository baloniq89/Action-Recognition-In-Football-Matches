import sys
import numpy as np
from matplotlib import pyplot as pl
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfile
from tkinter import *
from datetime import date, datetime
import visualkeras
from PIL import ImageFont
import PIL.Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ctypes import windll
import random

# from model import *
import customtkinter as ctk
from customtkinter import *

import tensorflow as tf
# print("Dostępne urządzenia GPU:")
# print(tf.config.list_physical_devices('GPU'))
# print("Wersja TensorFlow-GPU:", tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model
import os

windll.shcore.SetProcessDpiAwareness(1)
ctk.set_appearance_mode("dark")

# VGG16, ImageNet, ResNet, GoogLeNet


model = load_model('football_events_model_10_2024-01-05_10_17_12.h5')
print(model.history)
# class_mapping = ["Corner", "FreeKick", "KickOff", "Penalty", "RedCards", "Shoot", "Tackle", "ToSubstitue", "YellowCards",]
class_mapping = ["Corner", "FreeKick", "Penalty", "RedCards", "Tackle", "ToSubstitue", "YellowCards",]

# ---------------------------GUI----------------------------


root = ctk.CTk()
root.title("Football Situation Analizer")


root.geometry("800x600")
# root.update()
"Here all other widgetts will be added"

frame1 = ctk.CTkScrollableFrame(master=root, width=200, height=370, border_width=1, border_color="#FFCC70",  label_text="Menu")
frame1.grid(column=0, row=0)

frame2 = ctk.CTkFrame(master=root, width=200, height=470,border_width=1, border_color="#FFCC70",)
frame2.grid(column=1, row=0)
start_label_2 = ctk.CTkLabel(frame2, text="Load Image to see prediction")
start_label_2.pack(pady=40)

frame3 = ctk.CTkFrame(master=root, width=200, height=370,border_width=1, border_color="#FFCC70",)
frame3.grid(column=2, row=0)
start_label_3 = ctk.CTkLabel(frame3, text="Load Image to see predition chart")
start_label_3.pack(pady=40)

def clean_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def preprocess_image(img):
    img = img.resize((224, 224), PIL.Image.LANCZOS)
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def load_JPG():
    file = askopenfile(parent=frame1, mode='rb', title="Wybierz zdjecie")
    print(file)
    if file:
        text_box = tk.Label(frame2, text="Elegancko", font=("Raleway", 25))
        print("JPG loaded from the FILE Syste")
        img = PIL.Image.open(file)
        img_array = preprocess_image(img)
        predictions = model.predict([img_array])
        print(predictions)
        clean_frame(frame2)
        img2 = ctk.CTkImage(dark_image=img, size=(300, 300))
        my_label = ctk.CTkLabel(frame2, text="", image=img2)
        my_label.pack(pady=10)

        drawChart(predictions)
        class_index = np.argmax(predictions[0])

        text_box = ctk.CTkLabel(frame2, text=class_mapping[class_index], font=("Raleway", 25))
        text_box.pack(pady=40)

def load_model_JPG():
        clean_frame(frame2)
        clean_frame(frame3)
        img = ctk.CTkImage(dark_image=PIL.Image.open("model_1_structure.png"), size=(400, 400))
        my_label = ctk.CTkLabel(frame2, text="", image=img)
        my_label.pack(pady=10)

def generate_random_file_path(directory):
        
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print("Podana ścieżka nie istnieje lub nie jest folderem.")
        return None

    
    pliki_jpg = [plik for plik in os.listdir(directory) if plik.lower().endswith('.jpg')]

    if not pliki_jpg:
        print("Brak plików .jpg w folderze.")
        return None

    losowy_plik = random.choice(pliki_jpg)

    return os.path.join(directory, losowy_plik)

def load_random_image(directory='Test/test_data'):
        text_box = tk.Label(frame2, text="Elegancko", font=("Raleway", 25))
        print("JPG loaded from the FILE Syste")
        print(generate_random_file_path(directory=directory))
        img = PIL.Image.open(generate_random_file_path(directory=directory))
        img_array = preprocess_image(img)
        predictions = model.predict([img_array])
        print(predictions)
        clean_frame(frame2)
        img2 = ctk.CTkImage(dark_image=img, size=(300, 300))
        my_label = ctk.CTkLabel(frame2, text="", image=img2)
        my_label.pack(pady=10)

        drawChart(predictions)
        class_index = np.argmax(predictions[0])

        text_box = ctk.CTkLabel(frame2, text=class_mapping[class_index], font=("Raleway", 25))
        text_box.pack(pady=40)

def showHighestPredictions(predictions):
    dict = {}
    for x in predictions:
        for i in range(7):
            print(i)
            key = class_mapping[i]
            value = x[i] * 100
            dict[key] = value

    return dict

def drawChart(predictions):
    clean_frame(frame3)
    prediction_dict = showHighestPredictions(predictions)

    figure2, ax = plt.subplots(figsize=(5, 4), dpi=100)
    figure2.set_facecolor('white') 

    bars = ax.bar(prediction_dict.keys(), prediction_dict.values(), color=['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'black', 'gray'], width=0.6)

    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("Prediction %", fontsize=12)

    ax.set_title("Predictions for Each Class", fontsize=14)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis

    ax.legend(bars, prediction_dict.keys(), loc='center left', fontsize=5, bbox_to_anchor=(1, 0.5))

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45, ha='right', fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    chart = FigureCanvasTkAgg(figure2, frame3)
    chart.draw()
    chart.get_tk_widget().grid(row=0, column=2)

def _quit():
    root.quit()
    root.destroy()


browse_text = tk.StringVar()


jpg_button = ctk.CTkButton(master=frame1,textvariable=browse_text, text="Browse Image", corner_radius=32, fg_color="green", text_color="white",hover_color="#4158D0", border_color="#FFCC70", command=load_JPG)
browse_text.set("Browse Image")
jpg_button.pack(pady=20)
# jpg_button.grid(column=0, row=0, sticky='e')
# jpg_button.place(relx=0.5, rely=0.5, anchor="center")

model_load_button = ctk.CTkButton(master=frame1, text="Load Model", corner_radius=32, fg_color="green", text_color="white", command=load_model_JPG)
model_load_button.pack(pady=20)

jpg_random_button = ctk.CTkButton(master=frame1, text="Random Image", corner_radius=32, fg_color="green", text_color="white", command=load_random_image)
jpg_random_button.pack(pady=20)

quit_button = ctk.CTkButton(master=frame1, text="Quit", corner_radius=32, fg_color="red", command=_quit)
quit_button.pack(pady=20)

# quit_button.place(relx=0.5, rely=0.5, anchor="s")
# quit_button.grid(column=0, row=0)
# def run_GUI(root=root):
#     root.mainloop()
root.mainloop()