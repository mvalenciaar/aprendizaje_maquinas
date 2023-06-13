# aprendizaje_maquinas
Este repositorio se crea para fines académicos para la asignatura de aprendizaje de máquinas.

## **Trabajo Final de Aprendizaje de Máquinas**

### **Estudiantes**
- Daniel Chanci Restrepo: dchancir@unal.edu.co
- David Duque Uribe: daduqueu@unal.edu.co
- Maria Victoria Valencia Arango: mvalenciaar@unal.edu.co

### **Descripción del dataset:**
#### **Análisis de opinión sobre reseñas de películas / Sentiment Analysis on Movie Reviews**

Este dataset IMDB (Intenet Movie DataBase) contiene 50.000 reseñas de películas para el procesamiento de lenguaje natural y la analítica de texto; el dataset se encuentra disponible en la Universidad de Standford (link`http://ai.stanford.edu/~amaas/data/sentiment/`).

El Dataset presenta un reto de clasificación de sentimientos binario y está dividido de la siguiente manera:
- 25.000 reseñas para entrenamiento 
- 25.000 reseñas para testeo

Por recomendación de los autores, se sugiere que al momento de utilizar este dataset se referencie así:
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

La estructura de los datos, consta de dos columnas:
- `review` cuenta con tipo de dato String. Este contiene las reseñas de películas en idioma inglés.
- `sentiment` que contiene dos posibles valores: positive y negative.

**Limitantes:**
- Las reseñas deben estar en el idioma inglés, por lo que si se desea incluir reseñas en otro idioma, los modelos de estudio no lograrán obtener el aprendizaje deseado, ya que deben ser compatibles con el mismo idioma.

### **Metodología:**

Dentro de la dinámica de trabajo propuesto, se plantea una metodología así:
- Realizar exploración y limpieza de los datos asociados al análisis de sentimientos de reseñas de películas.
- Tokenizar las palabras de cada reseña, con el fin de que los modelos analíticos reciban entradas numéricas.
- Posterior a esta limpieza, se plantean 4 modelos  analíticos para el entrenamiento y predicción del sentimiento de las reseñas del dataset.
- Resultados de los 4 modelos analíticos y la selección del mejor accuracy obtenido entre ellos.
- Finalmente, concluir sobre los resultados obtenidos.

### **Presentación de resultados y uso de métricas:**

#### **Exploración y Limpieza del Dataset**

Para el procedimiento de exploración y limpieza del dataset, se realizaron las siguientes operaciones con la finalidad de obtener información de calidad que permitiera un mejor entrenamiento y predicción de los modelos analíticos planteados para el análisis de sentimientos de reseñas de películas.
* Conversión de respuesta de texto a binario.
* Eliminación de reseñas duplicadas.
* Evaluación de desbalanceo de datos.
* Eliminación de etiquetas HTML.
* Eliminación de caracteres especiales (URL, e-mail, corchetes, números)
* Convertir texto a minúscula.

Posteriormente, se obtienen 24.884 reseñas positivas (1), y 24.698 negativas (0).

![word_cloud](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/6c83a9c2-fbbc-466e-91db-400b6d27a6e8)


Figura 1. Nube de palabras

#### **Lematización**
Este proceso es una técnica del procesamiento de lenguaje natural (NLP) que tiene como objetivo, reducir palabras a su forma base o raíz; ejemplo: convertir plurales en singulares, verbos conjugados a infinitivos, entre otros.
La librería utilizada de Python para el proceso de remoción de Stopwords es NLTK.

A continuación, se muestra para nuestro dataset de trabajo como se genera esta transformación de lematización:


![Lematización](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/0db2cb42-abc1-47a5-9e96-5a1dcdc37daa)


Figura 2. Proceso de Lematización


#### **Remoción de Stopwords**

Esta técnica elimina palabras que no contienen significado alguno, por ejemplo: preposiciones, pronombres, conectores, entre otros.
La librería utilizada de Python para el proceso de remoción de Stopwords es `NLTK`.


![remove_stop_words](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/918da066-d211-4221-8caf-c71166cb488a)


Figura 3. Remoción de Stopwords

#### **Análisis de palabras comunes**

Después de aplicar los procesos de lenguaje natural a las reseñas, se considera pertinente revisar las palabras claves más comunes entre reseñas positivas y negativas y así entender los procesos que deberán realizar los métodos de Inteligencia Artificial propuestos. A continuación se muestran los resultados:

================================================Palabras más comunes de reseñas con sentimientos positivos =================================================

Word  | Count
------------- | -------------
film  | 48523
movie  | 42929
one  | 26843
see | 21185
make | 19755
like  | 19292
good | 17262
get | 15922
time | 15192
great | 13973

![palabras positivas más comunes](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/035250bb-aad3-43cb-bf44-ba4cc61cc695)

Figura 4. Palabras de reseñas positivas más comunes.

================================================Palabras más comunes de reseñas con sentimientos negativos =================================================

Word  | Count
------------- | -------------
movie  | 55342
film  | 42290
one  | 25431
make | 23257
like | 23140
bad  | 20604
get | 19094
see | 19080
good | 18150
even | 14812

![palabras negativas más comunes](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/8ee9fd02-05ce-4cc2-9f22-c96c29938651)

Figura 5. Palabras de reseñas negativas más comunes.

Cómo se puede ver mirando solo palabras comunes no se puede determinar de forma inmediata el sentimiento de una palabra contando palabras comunes debido a que ambos tipos de reseñas comparten palabras comunes (exceptuando 'bad'). Para esto, es conveniente revisar bigramas y trigramas; es decir, combinaciones de 2 y 3 palabras más comunes. El resultado se muestra en las siguientes gráficas:


================================================Bigramas más comunes de reseñas con sentimientos positivos =================================================

Word  | Count
------------- | -------------
(one, best) | 1608
(see, movie) | 1364
(watch, movie) | 1250
(ive, see) | 1056
(see, film) | 1039
(ever, see) | 1024
(even, though) | 1008
(look, like) | 916
(first, time) | 908
(dont, know) | 859

================================================Trigramas más comunes de reseñas con sentimientos positivos =================================================

Word  | Count
------------- | -------------
(ive, ever, see) | 345
(new, york, city) | 186
(film, ive, see) | 159
(one, best, movie) | 156
(world, war, ii) | 152
(film, ever, make) | 151
(movie, ever, see) | 142
(one, best, film) | 142
(movie, ive, see) | 134
(base, true, story) | 123

![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/167cca40-3bc2-45f6-be19-832157027e41)

![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/fb86cd25-c0b4-4148-8a0e-02c774b4c4ab)

Figura 6. Bigramas y Trigramas más comunes en reseñas positivas.

================================================Bigramas más comunes de reseñas con sentimientos negativos =================================================

Word  | Count
------------- | -------------
(look, like) | 2714
(bad, movie) | 2371
(ever, see) | 1751
(watch, movie) | 1711
(waste, time) | 1559
(special, effect) | 1406
(make, movie) | 1335
(see, movie) | 1260
(dont, know) | 1177
(bad, film) | 1124

================================================Trigramas más comunes de reseñas con sentimientos negativos =================================================

Word  | Count
------------- | -------------
(ive, ever, see) | 635
(bad, movie, ever) | 566
(movie, ever, see) | 378
(one, bad, movie) | 347
(dont, waste, time) | 307
(movie, ive, ever) | 270
(bad, film, ever) | 260
(bad, movie, ive) | 247
(movie, ever, make) | 229
(film, ever, see) | 191
 
![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/8e3ea13d-b283-4714-ad4a-b76102702796)

![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/7a423548-fabd-48a9-afe8-7e1c09f9f33e)

Figura 7. Bigramas y Trigramas más comunes en reseñas negativas.

De estas gráficas se puede concluir que hay formas de diferenciar los sentimientos de las reseñas mirando combinaciones de varias palabras, por lo cuál va a ser importante considerar la secuencilidad y la cantidad de veces que aparecen las palabras en las reseñas.

### **Tokenización**

La tokenización permite separar las oraciones en palabras, y a cada una de ellas se le asigna un valor numérico único, para permitir que los modelos tengan entradas numéricas.
Este se utiliza con la librería de Python `Keras`, y la función de esta librería `Tokenizer`, y recibe como fuente de datos el conjunto de entrenamiento.


<img width="635" alt="Tokenization" src="https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/0419b6d8-8669-4b31-bdff-80bf2991ec20">


Figura 8. Tokenización


Posteriormente, se tokeniza el conjunto de entrenamiento y validación, en donde se añade un `padding` para que todas las reseñas tengan la misma longitud al momento del procesamiento en el modelo analítico.


### **Análisis y conclusiones:**
### **Modelos de Aprendizaje Profundo**

Se considera pertinente aclarar que para ambos modelos presentados se va a incluir una capa llamada Embedding. Esta capa va a ayudar a crear una representación vectorial y semántica de las entradas del modelo (en este caso las reseñas tokenizadas) permitiendo también crear representaciones vectoriales similares de palabras con contexto semántico similar (Hombre y mujer, rey y reina). Esto es esencial para ambos modelos debido a que va a ser el insumo para los filtros de la capa convolucional de la red neuronal convolucional y para la red neuronal recurrente permitirá crear una representación vecotrial de la secuencialidad intrínseca dentro de las reseñas de películas. Más información sobre el proceso puede ser encontrada en el siguiente enlace https://machinelearningmastery.com/what-are-word-embeddings/

#### **Red Neuronal Convolucional**

Las Redes Neuronales Convolucionales son ampliamente usadas en el campo de detección de objetos y clasificación de imágenes, pero también pueden tener aplicaciones en el campo del procesamiento de lenguaje natural, bajo ciertas condiciones. La principal diferencia es el uso de una capa Convolucional con filtros de una dimensión, lo cuál sigue los mismos lineamientos de los filtros de una red convolucional pero con filtros en forma de vector unidimensional en vez de una matriz, y el resultado de este proceso será un espacio vectorial con las características más relevantes de la entrada ingresada en este caso rasgos semánticos relevantes detectados. Para el modelo propuesto se inicia como se mencionó anteriormente con una capa Embedding, seguido de una capa Convolucional unidimensional con un max pooling para la detección de atributos más relevantes. Se termina con una capa neuronal clásica con el resultado de la convolución y una función de activación sigmoide para devolver un resultado binario. Para determinar los mejores hiperparámetros se realiza una búsqueda exhaustiva usando GridSearchCV con resultados obtenidos como se muestra a continuación:

![CNN_Grid_Search](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/70264c33-68af-4604-aff6-0ef09957c06f)

Figura 9. Resultados De GridSearchCV para modelo CNN

Los mejores hiperparámetros que se obtuvieron fueron los siguientes: {'batch_size': 32, 'dropout_rate': 0.25, 'epochs': 10, 'filters': 64}, dónde el dropout rate hace referencia a la probabilidad de apagado de neuronas en la capa Dropout. El modelo fue entrenado durante 10 épocas obteniendo así una precisión en entrenamiento del 100%. Para el conjunto de evaluación, se obtuvo una precisión o accuracy de 0.8566 y un F1-score de 0.8591. A continuación se muestran los resultados gráficos obtenidos durante el entrenamiento y la matriz de confusión:

![CNN_training_curves](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/7accc4f0-05ae-4c43-b3d3-d19947588e65)

Figura 10. Curva de función de pérdida para modelo CNN

![CNN_cfm](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/adea593c-9ecd-480c-93aa-102b8b8d69f9)

Figura 11. Matriz de confusión para modelo CNN

De estas gráficas se puede ver que el modelo tiene buen rendimiento para entender la secuencialidad presentada en las reseñas de películas pero presenta un sobrentrenamiento notorio. Esto puede ser debido a que el modelo parece aprender los patrones de secuencialidad presentados por la tokenización en sólo dos épocas. Es posible que la información para el modelo sea fácil de procesar y entender gracias a la capa de Embedding añadida.

#### **Red Neuronal long short term memmory (LSTM)**

La Red Neuronal LSTM es un tipo de red neuronal recurrente, las cuales son muy útiles para modelar información secuencial como series de tiempo y lenguaje natural debido a que permite iterar sobre los pasos temporales de los datos secuenciales, manteniendo también un estado interno de lo aprendido durante el proceso.

La Red Neuronal LSTM ofrece la capacidad de guardar información que puede ser relevante para el contexto de la serie temporal y también determinar la información que puede ser despreciable.

![GridSearch_LSTM](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/5758c989-7998-4df8-a41f-bf6b89a48c39)

Figura 12. Resultados De GridSearchCV para modelo LSTM

Los mejores hiperparámetros que se obtuvieron fueron los siguientes: {'batch_size': 64, 'dropout_rate': 0.75, 'epochs': 5}, dónde el dropout rate hace referencia a la probabilidad de apagado de neuronas en la capa Dropout. Para el modelo LSTM se creó un modelo simple que recibe la información tokenizada a través de una capa Embedding, para seguir con una capa recurrente (LSTM) de 64 perceptrones. Se tienen también dos capas Dropout que ayudarán a disminuir el sobrentrenamiento del modelo. El Modelo fue entrenado durante 10 épocas (aunque solo realizó 7 debido al early stopping). En el conjunto de evaluación se obtuvo una precisión o accuracy de 0.8650 y un F1-score de 0.8672, y un 0.9432 de precisión obtenido en la fase de entrenamiento.

A continuación se presentan los resultados del entrenamiento y la matriz de confusión:

![lstm_loss_function](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/c3228d4f-ab8e-497f-a68e-cf27b0b4a4ba)

Figura 13. Curva de función de pérdida para modelo LSTM

![lstm_precision_curve](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/7b55c191-2757-43ff-bc10-a8570aa2b1a1)

Figura 14. Curva de precisión para modelo LSTM

![lstm_cfm](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/77431610-f956-42ba-b5d5-33dfbef71987)

Figura 15. Matriz de confusión LSTM

Al igual que el modelo convolucional se nota un sobrentrenamiento notorio que puede deberse a la regularización aplicada antes y después de la capa recurrente y que aprende rápido de los datos. Es posible que con más épocas de entrenamiento pueda darse una leve mejoría en la precisión. 

### **Modelos tradicionales de Aprendizaje de Máquinas**
#### **Modelo regresión logistica**

La regresión logistica es un tipo de analisis de regresión que busca predecir el resultado de una variable categorica (0 y 1) en función de las variables independientes o predictorias (reseñas).

Para el modelo de regresión logistica se realizó un grid search con el objetivo de optimizar los hiperparametros del modelo, donde se encontró que los mejores hiperparametros fueron: (C=10.0, penalty='l2', max_iter=500). Sin embargo, el desempeño del modelo no fue el más optimo puesto que presento una precisión o accuracy del 0.5223 y 0.5099 para los datos de entrenamiento y test, respectivamente. Además el modelo presento un valor del 0.6001 para el F1 score. A continuación se presenta la matriz de confusión para el modelo, ver Figura 16.

![lr_cfm](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/360707f9-6258-40af-aabf-6dc34e8538f0)

Figura 16. Matriz de confusión regresión logistica.

La curva ROC es una presentación grafica de la sensibilidad frente a la especificidad para un sistema de clasificación. Donde entre más cercana se encuentre la curva a la esquina superior izquierda, significa un mejor desempeño del modelo, caso contrario es equivalente a una clasificación 'aleatoria'. Como se observa en la Figura 17, el modelo se acerca a una clasificación aleatoria y no a una buena clasificación.

![curva_ROC_logistic_regresion](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/f27deed1-127e-430a-b815-b84c6f2199a7)

Figura 17. Curva ROC regresión logistica.

Finalmente, se concluye que el modelo de regresión logistica no presenta un buen desempeño para la calsificación de reseñas de peliculas.

#### **Modelo arboles de decisión**

Un arbol de decisión es un modelo de aprendizaje supervisado, que puede ser utilizado para modelos de clasificación. El cual tiene una estructura de arbol jerarquico, consta de nodo raiz, ramas, nodos internos y nodos hojas.

Para realizar el entrenamiento del arbol de decisión se realizó un grid search buscando optimizar los hiperparametros para un mejor desempeño del modelo. Sin embargo, luego de evaluar el modelo no es un modelo muy optimo y presenta un desempeño bajo con los siguientes valores: precisión o accuracy de 0.5460 0.5211ara los datos de entrenamiento y test, respectivamente, adicionalmente el valor F1 score fue de 0.4329. A continuación presenta la matriz de confusión, ver Figura 18, y la curva ROC, ver Figura 19, la cual nuevamente se aleja de la esquina superior izquiera (optimo) y se acerca a una clasificación 'aleatoria'.

![matriz_confusion_decision_tree](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/b14a4331-9a76-4590-8608-294d18f15c2a)

Figura 18. Matriz de confusión arbol de decisiones.

![curva_ROC_decision_tree](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/c2ccb89b-01e0-41c9-bc79-33d0a37ed2d8)

Figura 19. Curva ROC arbol de decisión.


#### **Mejora para Modelo de Regresión logística

Para mejorar la precisión de los modelos clásicos de aprendizaje de máquinas, se decide realizar un proceso de tokenización diferente al usado inicialmente, el cual consistirá en usar la función `CountVectorizer`. A diferencia de la función Tokenizer, CountVectorizer devuelve una representación vectorial donde cada elemento corresponde a la cantidad de veces que aparece una palabra obtenida del vocabulario de todos los documentos obtenidos, por ejemplo:

[[0 0 1 1 0 0 1 0 0 0 0 1 0]
 [0 1 0 1 0 1 0 1 0 0 1 0 0]
 [1 0 0 1 0 0 0 0 1 1 0 1 0]
 [0 0 1 0 1 0 1 0 0 0 0 0 1]]

Dónde cada fila de la matriz hace referencia a a un documento vectorizado. Los elementos de cada documento pueden ser binarios o valores mayor a 1 que representen mayor frecuencia. Después de aplicar el proceso de tokenización y entrenar el modelo, se obtienen los siguientes resultados (Ver figuras 20 y 21):

![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/f6f8d703-540e-4a64-80ce-1175362cc985)

Figura 20. Matriz de confusión de regresión logística mejorada

![image](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/32648633/10b6c07b-871e-4cc5-a3a6-fc82815ada2c)

Figura 21. Curva ROC de regresión logísitca mejorada.

Gráficamente se ve que se obtuvien muy buenos resultados con respecto al uso de la función Tokenizer para parametrizar el conjunto de datos. Con respecto a las métricas, se obtuvo una precisión en entrenamiento de 0.9999, en el conjunto de evaluación de 0.8723 y un F1-score de 0.8738. Esta mejoría puede explicarse a que los métodos clásicos de aprendizaje de máquinas esperan una entrada fija numérica, y no desarrollan inteés en aprender de procesos secuenciales (caso necesario para los métodos de aprendizaje profundo).


### **Conclusiones**

* Modelos de clasificación más simples, como el de regresión logistica y arbol de decisiones no presentaron un buen desempeño en la tarea de clasificación, lo cual se interpreta que es debido a la tecnica de tokenización usada, la cual funciona mejor para modelos de inteligencia artificial más avanzados. Puesto que el metodo utilizado conserva el orden en el que aparecen las palabras, lo cual puede cambiar el significado de una frase. Este tipo de tokenización es más optimo para redes neuronales que buscan patrones más profundos. Caso contrario sucede con modelos como arbol de decisiones y regresión logistica, los cuales al ser modelos lineales, no comprenden la importancia del orden de las palabras, y se centran más en la frecuencia de las mismas perdiendo datos y patrones importantes para la generalización del modelo.

* El tipo de tokenización usada para los datos de entrada de los modelos propuestos de inteligencia artificial debe depender de los requerimientos, la naturaleza de los datos y el objetivo que se trata de alcanzar/solucionar con el modelo. Como se puede ver la regresión logística es la opción más viable si se utiliza una técnica de tokenización CountVectorizer, la cuál le va a dar prioirdad a las frecuencias de las palabras en vez de secuencialidades entre estas. 
