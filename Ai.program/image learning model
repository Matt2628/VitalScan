# Imorterer resnet50:
import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # Resnet 50 er en forrhåndstrent dyp læringsmodell
from tensorflow.keras.applications.resnet50 import preprocess_input # importerer en funksjon som forbehandler bildene på måten som kreves av resnet 50
from tensorflow.keras.preprocessing.image import ImageDataGenerator # brukes for å forbehandle bilder fra mapper, inkludert agrumentering hvis ønsket
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D   # lag som legges oppå resnet: fullt tilkoblet lag og global gjennomsnittspooling
from tensorflow.keras.models import Model   # gjør det mulig å bygge opp en ny modell basert på eksisterende lag
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # gjør at modellen stopper automatisk når den ikke blir bedre, og lagrer den beste versjonen underveis


base_model = ResNet50(weights= 'imagenet', include_top=False, input_shape=(224, 224, 3)) # laster in resnet 50  med forhåndstrente vekter fra imageNet, uten det siste klassifikasjonslaget, bildene skal være 224 x 224 med 3 fargekanaler)

for layer in base_model.layers: # fryser de opprinnelige lagene i resnet slik at de ikke oppdateres under trening ( modellen brukes som en feature extractor)
    layer.trainable = False

x = base_model.output # henter utgangen fra resnet
x = GlobalAveragePooling2D()(x) # gir en kompakt representasjon ved å redusere dimensjonene (ta gjennomsnittet over hele bildeområdet)
x = Dense(128, activation='relu')(x)    # legger til et fullt tilkoblet lag med 128 noder og ReLU-aktivering (lærer nye mønstre)
predictions = Dense(4, activation='softmax')(x) #output-lag med 4 noder for 4 klasser og softmax aktivering som gir sannsynligheter for hver klasse

model = Model(inputs=base_model.input, outputs= predictions) # setter sammen hele modellen: input fra ResNet, output fra dine nye lag 

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
) # deler opp hver av de 4 mappene og gjør 80% av innholdet til treningsdata og 20% til testdata

train_generator = datagen.flow_from_directory(
    r"C:\Users\norun\OneDrive - Universitetet i Innlandet\Dokumenter\mri_scan_project\brain-tumor-mri-dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
) # laster treningsdata fra mappene, skalerer bildene til 224 x 224, grupperer bildene i batches på 32, bruker kategorisk klassifisering (en-hot encoding for 4 klasser)

test_generator = datagen.flow_from_directory(
    r"C:\Users\norun\OneDrive - Universitetet i Innlandet\Dokumenter\mri_scan_project\brain-tumor-mri-dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
) # samme som forrige men for testdataene


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # kompilerer modellen med Adam-optimalisering, kategorisk krysstap (for en-hot klassifisering), måler nøyaktigheten

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True)
] # Early stopping når modellen har gått gjennom antallet epoker nødvendig for å bli best mulig og lagring av modellen på sitt peak

model.fit(
    train_generator,
    validation_data= test_generator,
    epochs=30,
    callbacks=callbacks
    ) # trener modellen på treningsdata i 30 epoker, og evaluerer på testdata underveis

model.save('mri_analysis_model.keras') # lagrer ferdigtrent modell med keras

loss, acc = model.evaluate(test_generator) # evaluerer modellen på testdata returnerer tap og nøyaktighet
print(f"Test accuracy: {acc: .2f}") # skriver ut testnøyaktigheten med to desimaler
