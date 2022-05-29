from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from imutils import paths
import os

#Declaramos el path de de nuestro proyecto
mainPath = "/home/elio987/Documents/deteccion_senales_CNN"
#path del modelo a analizar
modelPath = os.path.join(mainPath, "modelos")
#path de los videos a analizar
videoPath = os.path.join(mainPath, "videos_resultantes")
#Tamano de la imagen necesaria
imageSize = (64, 64)
#Diccionario de las clases de la DATASET
classDictionary = {0: 'Speed Limit 20 km/h',1:'Speed Limit 30 km/h',2:'Speed Limit 50 km/h',3:'Speed Limit 60 km/h',
                   4:'Speed Limit 70 km/h',5:'Speed Limit 80 km/h',6:'End of Speed Limit 80 km/h',7:'Speed Limit 100 km/h',
                   8:'Speed Limit 120 km/h',9:'No passing',10:'No passing for vechiles over 3.5 metric tons',
                   11:'Right-of-way at the next intersection',12:'Priority road',13:'Yield',14:'Stop',15:'No vechiles',
                   16:'Vechiles over 3.5 metric tons prohibited',17:'No entry',18:'General caution',19:'Dangerous curve to the left',
                   20:'Dangerous curve to the right',21:'Double curve',22:'Bumpy road',23:'Slippery road',24:'Road narrows on the right',
                   25:'Road work',26:'Traffic signals',27:'Pedestrians',28:'Children crossing',29:'Bicycles crossing',
                   30:'Beware of ice/snow',31:'Wild animals crossing',32:'End of all speed and passing limits',33:'Turn right ahead',
                   34:'Turn left ahead',35:'Ahead only',36:'Go straight or right',37:'Go straight or left',38:'Keep right',39:'Keep left',
                   40:'Roundabout mandatory',41:'End of no passing',42:'End of no passing by vechiles over 3.5 metric tons'}
#Diccionario de las probabilidades obtenidas
classPercentage = {'Speed Limit 20 km/h':0,'Speed Limit 30 km/h':0,'Speed Limit 50 km/h':0,'Speed Limit 60 km/h':0,
                   'Speed Limit 70 km/h':0,'Speed Limit 80 km/h':0,'End of Speed Limit 80 km/h':0,'Speed Limit 100 km/h':0,
                   'Speed Limit 120 km/h':0,'No passing':0,'No passing for vechiles over 3.5 metric tons':0,
                   'Right-of-way at the next intersection':0,'Priority road':0,'Yield':0,'Stop':0,'No vechiles':0,
                   'Vechiles over 3.5 metric tons prohibited':0,'No entry':0,'General caution':0,'Dangerous curve to the left':0,
                   'Dangerous curve to the right':0,'Double curve':0,'Bumpy road':0,'Slippery road':0,'Road narrows on the right':0,
                   'Road work':0,'Traffic signals':0,'Pedestrians':0,'Children crossing':0,'Bicycles crossing':0,
                   'Beware of ice/snow':0,'Wild animals crossing':0,'End of all speed and passing limits':0,'Turn right ahead':0,
                   'Turn left ahead':0,'Ahead only':0,'Go straight or right':0,'Go straight or left':0,'Keep right':0,'Keep left':0,
                   'Roundabout mandatory':0,'End of no passing':0,'End of no passing by vechiles over 3.5 metric tons':0}
#Cargamos el modelo
model = load_model(os.path.join(modelPath, "signals_43_student.model"))
#Frames totales
frames_totales = 0

# Escribir el nombre del video que se quiere probar
vid = cv2.VideoCapture(os.path.join(videoPath, "siga.webm"))

while (vid.isOpened()):
    #Leemos la camara
    ret, image = vid.read()
    #Si el frame esta vacio terminamos el analisis
    if image is None:
        break
    else:
        output = image.copy()

    #Preprocesamos la imagen otra vez para poder analizarla
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, imageSize)
    image = image.astype("float") / 255.0

    #Expandimos sus dimensiones
    image = np.expand_dims(image, axis=0)

    #Hacemos la prediccion
    predictions = model.predict(image)
    #Obtenemos la maxima probabilidad
    classIndex = predictions.argmax(axis=1)[0]
    #Obtenemos el label de la prediccion
    label = classDictionary[classIndex]
    prob = "{:.2f}%".format(round(predictions[0][classIndex] * 100,2))
    #Sumamos los frames detectados de cada clase
    classPercentage[label] = classPercentage[label]+1
    label = label + " " + prob
    "Contamo todos los frames"
    frames_totales +=1
    textColor = (155, 5, 170)
    #Mostramos en cada frame del video la prediccion
    cv2.putText(output, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2)
    if (ret == True):
        cv2.imshow("img", output)
        if (cv2.waitKey(30) == ord('s')):
            break
    else:
        break
print("Porcentaje total obtenido: ")
for key in classPercentage.keys():
    classPercentage[key] = str(round((classPercentage[key]/frames_totales)*100,2))+" %"
print(classPercentage)

vid.release()
cv2.destroyAllWindows()
