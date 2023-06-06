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


![Screen Shot 2023-06-02 at 9 34 37 PM](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/0db2cb42-abc1-47a5-9e96-5a1dcdc37daa)


Figura 2. Proceso de Lematización


### **Remoción de Stopwords**

Esta técnica elimina palabras que no contienen significado alguno, por ejemplo: preposiciones, pronombres, conectores, entre otros.
La librería utilizada de Python para el proceso de remoción de Stopwords es `NLTK`.


![remove_stop_words](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/918da066-d211-4221-8caf-c71166cb488a)


Figura 3. Remoción de Stopwords


### **Tokenización**

La tokenización permite separar las oraciones en palabras, y a cada una de ellas se le asigna un valor numérico único, para permitir que los modelos tengan entradas numéricas.
Este se utiliza con la librería de Python `Keras`, y la función de esta librería `Tokenizer`, y recibe como fuente de datos el conjunto de entrenamiento.


<img width="635" alt="Tokenization" src="https://github.com/mvalenciaar/aprendizaje_maquinas/assets/56141354/0419b6d8-8669-4b31-bdff-80bf2991ec20">


Figura 4. Tokenización


Posteriormente, se tokeniza el conjunto de entrenamiento y validación, en donde se añade un `padding` para que todas las reseñas tengan la misma longitud al momento del procesamiento en el modelo analítico.



### **Análisis y conclusiones:**

#### **Modelo regresión logistica**

La regresión logistica es un tipo de analisis de regresión que busca predecir el resultado de una variable categorica (0 y 1) en función de las variables independientes o predictorias (reseñas).

Para el modelo de regresión logistica se realizó un grid search con el objetivo de optimizar los hiperparametros del modelo, donde se encontró que los mejores hiperparametros fueron: (C=10.0, penalty='l2', max_iter=500). Sin embargo, el desempeño del modelo no fue el más optimo puesto que presento una precisión o accuracy del 0.535 y 0.512 para los datos de entrenamiento y test, respectivamente. Además el modelo presento un valor del 0.562 para el F1 score. A continuación se presenta la matriz de confusión para el modelo, ver Figura 5.


![lr_cfm](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/103476375/032dd4fa-5448-4fcf-8581-4f81017ea4da)


Figura 5. Matriz de confusión regresión logistica.

La curva ROC es una presentación grafica de la sensibilidad frente a la especificidad para un sistema de clasificación. Donde entre más cercana se encuentre la curva a la esquina superior izquierda, significa un mejor desempeño del modelo, caso contrario es equivalente a una clasificación 'aleatoria'. Como se observa en la Figura 6, el modelo se acerca a una clasificación aleatoria y no a una buena clasificación.


![curva_ROC_logistic_rehresion](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/103476375/2dbc6f9f-d535-4fee-955c-f60aff64e948)


Figura 6. Curva ROC regresión logistica.

Finalmente, se concluye que el modelo de regresión logistica no presenta un buen desempeño para la calsificación de reseñas de peliculas.

#### **Modelo arboles de decisión**

Un arbol de decisión es un modelo de aprendizaje supervisado, que puede ser utilizado para modelos de clasificación. El cual tiene una estructura de arbol jerarquico, consta de nodo raiz, ramas, nodos internos y nodos hojas.

Para realizar el entrenamiento del arbol de decisión se realizó un grid search buscando optimizar los hiperparametros para un mejor desempeño del modelo. Sin embargo, luego de evaluar el modelo no es un modelo muy optimo y presenta un desempeño bajo con los siguientes valores: precisión o accuracy de 0.543 y 0.523 para los datos de entrenamiento y test, respectivamente, adicionalmente el valor F1 score fue de 0.486. A continuación se presenta la matriz de confusión, ver Figura 7, y la curva ROC, ver Figura 8, la cual nuevamente se aleja de la esquina superior izquiera (optimo) y se acerca a una clasificación 'aleatoria'.


![matriz_confusion_decision_tree](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/103476375/b494c0b6-0e18-4e28-9181-d36b44b174ee)


Figura 7. Matriz de confusión arbol de decisiones.


![curva_ROC_decision_tree](https://github.com/mvalenciaar/aprendizaje_maquinas/assets/103476375/eb349290-3c0e-4989-a5f8-7491315de0b2)


Figura 8. Curva ROC arbol de decisión.

### **Conclusiones**

* Modelos de clasificación más simples, como el de regresión logistica y arbol de decisiones no presentaron un buen desempeño en la tarea de clasificación, lo cual se interpreta que es debido a la tecnica de tokenización usada, la cual funciona mejor para modelos de machine learning más avanzados. Puesto que el metodo utilizado conserva el orden en el que aparecen las palabras, lo cual puede cambiar el significado de una frase. Este tipo de tokenización es más optimo para redes neuronales que buscan patrones más profundos. Caso contrario sucede con modelos como arbol de decisiones y regresión logistica, los cuales al ser modelos lineales, no comprenden la importancia del orden de las palabras, y se centran más en la frecuencia de las mismas perdiendo datos y patrones importantes para la generalización del modelo.

* 
