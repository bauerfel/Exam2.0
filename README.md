# ImageClassificationExam

## Problem Statement

### Endbenutzer:
Personen, die sich in Quallengebieten aufhalten und überprüfen möchten, ob die Quallen in ihrer Nähe gefährlich sind.

### Ziel der Endbenutzer:
Die Endbenutzer möchten sicherstellen, dass sie nicht durch gefährliche Quallen verletzt werden oder im schlimmsten Fall sterben.

### Zu lösendes Hindernis:
Die Endbenutzer stehen vor der Herausforderung, die richtigen Quallenarten schnell und genau zu identifizieren, um mögliche Gefahren zu vermeiden.

## Data Collection and Augmentation 

### Images collected
Jellyfish Dataset von Kaggle:
https://www.kaggle.com/datasets/anshtanwar/jellyfish-types

### Description of splitting images into classes/labelling images 

ich habe die Daten im Ordner JellyFish in den Unterordner train und val abgelegt. 
Es wurden jeweils unterschiedliche Bilder zu den Folgenden Klassen angeordnet.

- barrel_jellyfish

- blue_jellyfish

- compass_jellyfish

- lions_mane_jellyfish

- mauve_stinger_jellyfish

- Moon_jellyfish
### Data Augmentation description  
Um mehr Nutzen aus den vorhandenen Daten zu ziehen werden die Daten aufbereitet.

  batch_size=12,
  image_size=(150, 150)

Die Augmentation horizontal Flip die Bilder wilkürlich und die random Rotation von 0.2 dreht das Bild um die Bilder unterschiedlich Darzustellen um die Trainingsdaten zu vervielfältigen. 

  augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
]

## Model Training

### Data augmentation (code)

### Model Training: From scratch or transfer learning, fine tuning (code)

### Comparison of performance (code / text) 
Ich habe drei Modelle als grundvoraussetzung verwendet.Ich habe die Modelle InceptionResNetV2, EfficentNetB3 Modell und Xception

Vergleich:
 Die Grundvoraussetzungen sind genau gleich. (selber Datensatz usw.)

#### InceptionResNetV2 Modell
Das InceptionResNetV2 Modell hat eine deutlich höhere Accuracy als das EfficentNetB3 Modell erreicht. Die Accuracy war bei 83 Prozent.
#### EfficentNetB3 Modell
Dieses Modell ist mit abstand das ungenauste weil es nur eine Accuracy von 25 Prozent hat.
### Xception Modell
Dieses Modell hat die Beste Accuracy erreicht, mit einem Wert von 94.44 Prozent.

### Fazit 
das Letzte Modell Xception hat am besten abgeschnitten, deshalb wird auch dieses Modell im Frontend verwendet.

## Model Application

### Frontend and Backend
app.py Code:

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
 
labels = ['barrel_jellyfish','blue_jellyfish','compass_jellyfish','lions_mane_jellyfish','mauve_stinger_jellyfish','Moon_jellyfish']
 
def predict_jellyfish_type(uploaded_file):
    
    if uploaded_file is None:
        return "No file uploaded."
   
    model = tf.keras.models.load_model('Jellyfish_transferlearning.keras')
    # Load the image from the file path
    with Image.open(uploaded_file) as img:
        img = img.resize((150, 150)).convert('RGB')  # Convert image to RGB
        img_array = np.array(img)
        
 
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        confidences = {labels[i]: np.round(float(prediction[0][i]), 2) for i in range(len(labels))}
 
        return confidences
 
 
// # Define the Gradio interface
iface = gr.Interface(
    fn=predict_jellyfish_type,  # Function to process the input
    inputs=gr.File(label="Upload File"),  # File upload widget
    outputs="text",  # Output type
    title="Jellyfish Classifier",  # Title of the interface
    examples=["images/barrel_jellyfish.jpeg", "images/blue_jellyfish.jpeg", "images/compass_jellyfish.jpeg", "images/lions_mane_jellyfish.jpeg", "images/mauve_stinger_jellyfish.jpeg", "images/Moon_jellyfish.jpeg"],   
    description="Upload a picture of a Jellyfish (barrel Gefährlichkeit: Niedrig, blue Gefährlichkeit: Moderat, compass Gefährlichkeit: Moderat, lions mane Gefährlichkeit: Hoch, mauve stinger Gefährlichkeit: Moderat bis Hoch, Moon Gefährlichkeit: Niedrig) "  # Description of the interface
)
 
// # Launch the interface
iface.launch()

### Demo
https://huggingface.co/spaces/bauerfel/JellyFish

### Result of user validation 
Aufgrund fehlender Quallen in meiner Umgebung wurden Bilder von Quallen aus dem Internet verwendet. 

Result of User Validation: JellyFish Erkenner

Übersicht und Zweck:
Das Ziel dieser Benutzervalidierung war es, die Benutzerfreundlichkeit und Genauigkeit des JellyFish-Erkenners zu bewerten, der auf Hugging Face gehostet wird. Die Ergebnisse sollen als Grundlage für weitere Verbesserungen des Erkennungssystems dienen.

Testmethodik:
Die Validierung wurde mit 2 Teilnehmern durchgeführt, die verschiedene Quallenbilder hochgeladen und die Ergebnisse des Erkennungssystems bewertet haben. Jeder Teilnehmer hatte die Aufgabe, mindestens 5 Bilder zu testen und Feedback zu geben.

Teilnehmer:
Die Teilnehmer waren zwischen 24 und 25 Jahre alt, mit unterschiedlichem technischen Hintergrund:

1 Anfänger
1 mittelmässig erfahren
Durchgeführte Tests:

Testfälle:

Hochladen von Quallenbildern
Überprüfen der Erkennungsergebnisse
Bewertung der Genauigkeit
Feedback zu Benutzerfreundlichkeit
Ergebnisse:

Jeder Teilnehmer testete das System mit 5 unterschiedlichen Bildern von Quallenarten.
Benutzerfeedback:

Qualitatives Feedback:

Insgesamt wurde das Interface als intuitiv empfunden, jedoch gab es Schwierigkeiten beim Hochladen grosser Bilddateien. Die Erkennungsgeschwindigkeit war gut, aber die Genauigkeit variierte je nach Bildqualität.
Fragebögen:

0% der Benutzer fanden die Erkennungsergebnisse zufriedenstellend.
100% gaben an, dass sie das Tool regelmässig nutzen würden, wenn die Genauigkeit verbessert wird.
Analyse und Interpretation:

Erkennungsergebnisse:

Die Erkennungsgenauigkeit lag bei durchschnittlich 75%. Fehler traten hauptsächlich bei Bildern auf, die schlechte Beleuchtung oder eine geringe Auflösung hatten.
Benutzerfreundlichkeit:

Das Interface wurde als einfach und benutzerfreundlich bewertet. Probleme traten beim Hochladen grosser Dateien auf, was zu Verzögerungen und gelegentlichen Fehlermeldungen führte.
Empfehlungen:
Erkennung verbessern:
Optimierung des Erkennungsalgorithmus für bessere Ergebnisse bei niedrigerer Bildqualität.
Bildupload:
Verbesserung des Bilduploads, um auch grosse Dateien ohne Verzögerungen zu unterstützen.
Feedbackmechanismus:
Implementierung eines Mechanismus, um Benutzerfeedback direkt nach jedem Erkennungsvorgang zu sammeln.

