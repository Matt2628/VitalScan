

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocessing_input

# Output fil
from output import translate_prediction 

# Laster ned ferdig lagd modell
model = tf.keras.models.load_model("alzheimer_model.h5")

# Sett in test bilde
image_path = r #"Sett in hva du vil"

# Behandle bilde
img = image.load_img(image_path, taget_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocessing_input(img_array)

# Kjør modellen
pred = model.predict(img_array)

# Oversettelse
main_burden, class_probs, burden_probs = translate_prediction(pred)

# Gjør at hvert av stadiene kommer i ønsket rekkefølge
order = ["No evidence of disease (NED)",
         "Low tumor burden",
         "Moderate tumor burden",
         "High tumor burden (Advanced-stage cancer)"]

# Gjør det mulig å skrive inn pasient ID og dato:
patient_name = input("Enter full name of patient: ")
examination_date = input("Enter examination date (YYYY-MM-DD): ")

print("|------------------------------------------------------------|")
print("|              MRI BRAIN - DIAGNOSTIC SUMMARY                |")
print("|------------------------------------------------------------|")
print(f"| Patient Name: {patient_name}".ljust(60) + "|")                              
print(f"| Examination Date: {examination_date}".ljust(60) + "|")
print("| Diagnostic Software: VITALScan                             |")                                 
print("| Imaging Modality: MRI Brain                                |")
print("|------------------------------------------------------------|")
print("| DIAGNOSTIC BURDEN DISTRIBUTION                             |")
print("|                                                            |")
print("| Tumor Burden Assesment:")

for label in order: 
    line = f"| • {label}: {burden_probs[label]:.1f}%"
    print(line.ljust(60) + "|")

print("|                                                            |")
print("| Overall clinical impression:                               |")
print(f"| {main_burden}".ljust(60) + "|")
print("|------------------------------------------------------------|")
