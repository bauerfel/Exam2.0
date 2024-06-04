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
Ich habe zwei Modelle als grundvoraussetzung verwendet. zum einen das InceptionResNetV2 Modell zum anderen das # EfficentNetB3 Modell

Vergleich:
Als Die Krundvoraussetzungen sind genau gleich. (selber Datensatz usw.)

#### InceptionResNetV2 Modell
Das InceptionResNetV2 Modell hat eine deutlich höhere Accuracy erreicht. Sie erreicht ein genauigkeit von 83 Prozent, was star
#### EfficentNetB3 Modell