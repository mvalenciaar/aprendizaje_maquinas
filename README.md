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
