
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Output fil
from translated_data import translate_prediction 

# Laster ned ferdig lagd modell
model = tf.keras.models.load_model("alzheimer_model.h5")

# Sett in test bilde
image_path = r"C:\Users\rosma\Documents\Maskinlæring II - Programmering\Python - Vår Semester 2026\Gruppe_Oppgave_1\Prosjekt\Testbilder\41598_2023_41576_Fig1_HTML.jpg"

# Behandle bilde
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Kjør modellen
pred = model.predict(img_array)

# Oversettelse
main_burden, burden_probs = translate_prediction(pred)

# Gjør at hvert av stadiene kommer i ønsket rekkefølge
order = ["No evidence of disease (NED)",
         "Low tumor burden",
         "Moderate tumor burden",
         "High tumor burden (Advanced-stage cancer)"]

# Gjør det mulig å skrive inn pasient ID og dato:
patient_name = input("Enter full name of patient: ")
examination_date = input("Enter examination date (YYYY-MM-DD): ")

print("|------------------------------------------------------------|")
print("|                                                            |")
print("|              MRI BRAIN - DIAGNOSTIC SUMMARY                |")
print("|                                                            |")
print("|------------------------------------------------------------|")
print("|                                                            |")
print("| PATIENT AND EXAMINATION DETAILS                            |")
print("|                                                            |")
print(f"| Patient Name: {patient_name}".ljust(61) + "|")                              
print(f"| Examination Date: {examination_date}".ljust(61) + "|")
print("| Diagnostic Software: VITALScan                             |")                                 
print("| Imaging Modality: MRI Brain                                |")
print("|------------------------------------------------------------|")
print("|                                                            |")
print("| DIAGNOSTIC BURDEN DISTRIBUTION                             |")
print("|                                                            |")
print("| Tumor Burden Assesment:                                    |")

for label in order: 
    line = f"| • {label}: {burden_probs[label]:.1f}%"
    print(line.ljust(61) + "|")
print("|------------------------------------------------------------|")
print("|                                                            |")
print("| OVERALL CLINICAL IMPRESSION:                               |")
print(f"| {main_burden}".ljust(61) + "|")
print("|------------------------------------------------------------|")
