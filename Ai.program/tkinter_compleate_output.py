
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from translated_data import translate_prediction

import tkinter as tk
from tkinter import font

def show_report(report_text: str):
    window = tk.Tk()
    window.title("VITALScan")

    mono = font.Font(family="Courier", size=11)

    text_widget = tk.Text(window, font=mono, width=80, height=40, bg="white")
    text_widget.pack(padx=10, pady=10)

    text_widget.insert("1.0", report_text)
    text_widget.config(state="disabled")

    window.mainloop()

model = tf.keras.models.load_model("mri_analysis_model.keras")

image_path = r"C:\Users\rosma\Documents\Maskinlæring II - Programmering\Python - Vår Semester 2026\Gruppe_Oppgave_1\Prosjekt\Testbilder\no-0002.jpg"

img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

pred = model.predict(img_array)

main_burden, burden_probs = translate_prediction(pred)

order = ["No evidence of disease (NED)",
         "Low tumor burden",
         "Moderate tumor burden",
         "High tumor burden (Advanced-stage cancer)"]

# Gjør det mulig å skrive inn pasient ID og dato:
print()
patient_name = input("Enter full name of patient: ")
examination_date = input("Enter examination date (YYYY-MM-DD): ")
print()

lines = []
lines.append("|------------------------------------------------------------|")
lines.append("|                                                            |")
lines.append("|               MRI BRAIN - DIAGNOSTIC SUMMARY               |")
lines.append("|                                                            |")
lines.append("|------------------------------------------------------------|")
lines.append("|                                                            |")
lines.append("| PATIENT AND EXAMINATION DETAILS                            |")
lines.append("|                                                            |")
lines.append(f"| Patient Name: {patient_name}".ljust(61) + "|")                              
lines.append(f"| Examination Date: {examination_date}".ljust(61) + "|")
lines.append("| Diagnostic Software: VITALScan                             |")                                 
lines.append("| Imaging Modality: MRI Brain                                |")
lines.append("|------------------------------------------------------------|")
lines.append("|                                                            |")
lines.append("| DIAGNOSTIC BURDEN DISTRIBUTION                             |")
lines.append("|                                                            |")
lines.append("| Tumor Stage:                                               |")

for label in order: 
    line = f"|  • {label}: {burden_probs[label]:.1f}%"
    lines.append(line.ljust(61) + "|")

lines.append("|------------------------------------------------------------|")
lines.append("|                                                            |")
lines.append("| OVERALL CLINICAL IMPRESSION                                |")
lines.append("|                                                            |")
lines.append(f"| {main_burden}".ljust(61) + "|")
lines.append("|------------------------------------------------------------|")

report_str = "\n".join(lines)

show_report(report_str)
