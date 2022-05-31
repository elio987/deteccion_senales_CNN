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
classDictionary = {0: "stop", 1: "fin_prob", 2: "derecha", 3: "izquierda", 4: "siga", 5: "rotonda"}
#Diccionario de las probabilidades obtenidas
classPercentage = {"stop":0,"fin_prob":0, "derecha":0, "izquierda":0,"siga":0,"rotonda":0}
#Cargamos el modelo
model = load_model(os.path.join(modelPath, "signals_5_student.model"))
#Frames totales
frames_totales = 0

# Escribir el nombre del video que se quiere probar
vid = cv2.VideoCapture(os.path.join(videoPath, "izq.webm"))

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
