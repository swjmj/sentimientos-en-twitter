# the_thing_of_the_words

Este es un codigo prueba para un clasificador de positividad o negatividad en un conjunto de textos,
su principal proposito es clasificar si un tweet es positivo o negativo.

El modelo es entrenado usando el general corpus (2012) proporcionado por http://tass.sepln.org/.


Los tweets a classificar se deben proporcionar en formato csv.


Las herramientas con las que se disponen son:

1.- Extrer tweets: Extrae los tweets que se utilizaran para el entranamiento del modelo, estos se ingresan en formato xml.

2.- Deteccion del idioma de los tweets. Esto es útil cuando se hace scrapping y los tweets se encuentran en un idioma distinto al español.
El algoritmo hace uso de tres librerías para detectar el idioma: langid, langdetect y TextBlob con el método detect_lenguage. Finalmente los tweets se guardan en un archivo csv con los resultados de las pruebas. Si es necesario el usuario ahora puede filtrar los tweets que no pasen una, dos o las tres pruebas.

3.- Con esta opción el programa hace una busqueda en malla de los mejores parámetros para entrenar una SVM utilizando la función GridSearchCV de sklearn. Nota de manera automatica el programa aplica funciones de limpieza de texto: stemming, tokenizing, tagging, remueve stopwords en español, etc... Cuando termine este proceso y el entrenamiento del SVM se imprimiran los mejores parámetros encontrados y se guardarán en el archivo best_params.csv.

4.- Se hace un entrenamiento con los mejores parámetros encontrados en la opción anterior e imprimer el f1 score y el coeficiente de correlación de Matthews. Finalmente se guarda el modelo para futuro uso.

5.- Predicción: En está opción se carga el modelo entrenado anteriormente y se puede ingresar una frase para hacer una predicción acerca de su positividad o negatividad.


Cosas a tomar en cuenta: 
Hay varios paths que estan escritos a mano en el codigo fuente, en especial los nombres de los archivos que se leen y guardan. El flujo de trabajo sería:
  Opción 1 (aquí no es necesario renombrar nada) 
  Opción 2 (opcional)
  Opción 3 (en general esto es automatico, pero cuando se quieren hacer pruebas puede ser muy engorroso estar cambiando el       archivo donde se guardan las cosas en el codigo fuente).
  Opción 4.- En esta opción no se tiene que hacer mucho a menos que se quiera cambiar el nombre del archivo donde se guarda el modelo.
  Opción 5.- Sólo queda probar el modelo de manera manual.
  


Cualquier comentario es bienvenido.
