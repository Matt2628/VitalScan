import tensorflow as tf
from tensorflow.keras.models import load_model # importerer den ferdigtrente modellen "alzheimer_model.h5"
from tensorflow.keras.preprocessing import image # importerer funksjon som laster inn og konverterer bildene til munpy arrays
from tensorflow.keras.applications.resnet50 import preprocess_input # tilpasser bilder slik at det får riktig format og skale før det sendel inn i modellen
import numpy as np
import sys # brukes for å lese inn argumenter fra kommandolinjen ( filstien til bildet som bruker skal teste)

#model = load_model(r"C:\Users\norun\OneDrive - Universitetet i Innlandet\Dokumenter\Maskinlæring\Maskinlæring_2\AlzheimerProject\alzheimer_model.h5") # laster inn den ferdigtrente modellen
model = tf.keras.models.load_model("mri_analysis_model.keras") # laster inn den ferdigtrente modellen

img_path = sys.argv[1] # henter bildet fra kommandolinjen

img = image.load_img(img_path, target_size=(224, 224)) # laster inn bilder fra img_pth , skalerer bildet til 224 x 224 piksler (alle bilder må skaleres til denne størrelsen for å få riktig input format, fordi ResNet50 er trent på bilder i denne størrelsen)
img_array = image.img_to_array(img) # konverterer bilder til numpy array slik at modellen kan forstå formatet
img_array = np.expand_dims(img_array, axis=0) #legger til en ekstra dimensjon i starten av arrayet fordi modellen forventer inputpå en spesiell form
img_array = preprocess_input(img_array) # forbehandler bildet med resnet sin preprocess_input, tilpasser pikselverdiene ti samme skala og format modellen ble trent på

pred = model.predict(img_array) # sender bildet gjennom modellen og får ut en prediksjon
class_names = ['Glioma', 'Meningioma', 'NoTumor', 'Pituitary'] # klassenavnene i samme rekkefølge som modellens output

print("Prediksjon:", class_names[np.argmax(pred)]) # finner indeksen til den høyeste sannsyneligheten, henter klassenavnet på den indeksen, skriver ut resultatet i terminalen


# skriv inn i terminalen: python test_mri_model.py "pathen i filsystemet til bildet man skal teste"
