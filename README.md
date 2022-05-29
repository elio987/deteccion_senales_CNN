# deteccion_senales_CNN
Deteccion de senales de trafico con el uso de CNN, una red detecta 6 senales y otra detecta 43.
- Este proyecto fue entrenado en google colab, para hacer el tiempo de entrenamiento mas pequeño.
- Puedes entrenar dos detectores de diferentes magnitudes, uno que detecte solamente 6 señales o uno que detecte 43 señales, el de 6 señales lo puedes entrenar con la RAM que tiene disponible google colab en su versión gratuita, si ya quieres hacer el completo necesitas usar por lo menos la versión pro.
- Estos son los enlaces a la DATASET ya subida en Drive:
- DATASET 6 señales:
https://drive.google.com/drive/folders/1-3hc9RzfkL2eOIG2YRX-q0-WGxtW_DMA?usp=sharing
- DATASET 43 señales:
https://drive.google.com/drive/folders/1fA9DtcOlEPpT_WcOAAgDyHBUZyQvRG0R?usp=sharing
- Enlace a la DATASET original:
https://benchmark.ini.rub.de/gtsrb_news.html
- En una carpeta llamda training va a estar el notebook para poder entrenar el modelo.
- En la carpeta llamada testing va a estar el archivo de python para poder detectar las imagenes con la webcam de su computadora.
- También en la carpeta d testing va a estar el codigo para poder validad la calidad de la deteccion de imagenes.
- En este ejemplo de uso para entrenar el modelo tensorflow-gpu==2.4.0
Al correr el testing se va a abrir tu camara con cv2 y mostrara en la imagen un texto diciendo que señal pertenece cada frame.
