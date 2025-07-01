# Asistente de IA: Aplicación RAG sobre temario de Certificación de Apache Spark Databricks.

Este repositorio contiene un proyecto completo para construir y evaluar una aplicación de **Retrieval-Augmented Generation (RAG)** que ayuda a preparar el examen **Databricks Certified Associate Developer for Apache Spark**.

---
## Objetivo del Proyecto

Diseñar un asistente inteligente que responda preguntas sobre el temario oficial del examen, usando un sistema RAG basado en LlamaIndex y modelos OpenAI. La fuente de conocimiento es el libro de estudio oficial en PDF.

---

## Estructura del proyecto. 

+ `data/` : contiene la fuente de datos (fichero .pdf) y los archivos generados por los estudios realizados en el proyecto.
+ `notebooks/`: contiene los notebooks enumerados según la realicación del proyecto, finalizando con la aplicación final RAG.
+ `scripts/`: contiene un script .py con todas las funciones definidas y documentadas que se utilizan a lo largo del proyecto.
+ `requirements.txt`: Dependencias del entorno

## Fases del proyecto. 

1. **Preprocesamiento del libro.**

+   Se ha recortado el fichero pdf, ubicado en `data/raw_text/`, de forma que se leen y aplanan las páginas correspondientes a la teoría que el modelo LLM debe utilizar para alimentarse.
+   Se procesa y limpia el texto para que sea legible y fácil de comprender por el modelo al ser utilizado como contexto para la respuesta a preguntas del usuario.
+   Una vez se tiene el texto plano limpio, se exporta en `data/plain_text/`

3. **Indexación y Vectorización: Estudio de chunk_size y chunk_overlap óptimos según la similitud del coseno.**

+   Se prueban distintos tamaños de chunks y solapamiento para que, con la creación de un retriever para cada uno de ellos, se eliga el tamaño adecuado que proporciona mejores métricas de similitud. 
+   Para ello, se definen funciones y se generan un excel con la métrica de los nodos devueltos por el retriever, el cual se encuentra en la ruta `data/scores_retriever/`. 
+   Una vez se selecciona el que más sentido y mejor métrica obtenga, se exporta el Index Storage en la ruta `data/index_storage/`.

4. **Evaluación Manual con preguntas de usuario.**

+   Se realizan un banco de preguntas predefinidas para comprobar el comportamiento del modelo LLM.
+   Se definen y afinan prompts de sistema y de mensaje de forma que se limite al LLM al contexto pasado por el retriever.
+   Se validan manualmente los chunks recuperados y la respuesta del modelo. 

5. **Evaluación sobre los Test que recoge el temario.** Para esta parte del proyecto se definen dos procesos y se obtiene la aplicación final, dando lugar a 3 notebooks.

+   `notebooks/04_extract_exams.ipynb`: Proceso similar a la fase 1, se lee el pdf fuente recortado sólo a las páginas con los test de estudio que proporciona el temario. La limpieza y procesado de esta parte es distinta, puesto que se finaliza teniendo una lista de prguntas procesadas como un diccionario de python las cuáles recogen entre otras cosas el test al que pertenece la pregunta, la respuesta correcta a dicha pregunta, y un ID identificativo para la comprobación manual del correcto procesado de las mismas.
+   `notebooks/05_Test2_exams.ipynb`: Este notebook sirve para la evaluación manual del comportamiento de la aplicación RAG al someterla a preguntas de examen. En él, se afinan los prompts para que el modelo no alucine con información no recuperada por el retriever en el contexto, y para que intente justificar con los nodos de contexto la elección de la respuesta final, de esta manera se puede evaluar el comportamiento de la aplicación y si su delimitación al contexto ha sido correcta.
+   `notebooks/06_FINAL_Simplified_RAG.ipynb`: Es el notebook final con los prompts y los chunks considerados óptimos para el objetivo del proyecto. Esta aplicación final será expuesta a un Test completo del libro para así evaluar la tasa de acierto. 

## Consideraciones. 

La aplicación se basa en la información recuperada de los nodos para responder preguntas, no utiliza información previa que pueda "conocer" el modelo LLM utilizado. Con esto, se entiende que algunas preguntas de razonar no sean respondidas correctamente por el modelo, de igual forma se le permite razonar en base a los chunks devueltos. 

Se consideró realizar la separación de chunks utilizando el índice del libro para aportar información sobre meta-datos de cada chunk. Finalmente, se descartó porque no se consiguieron mejores métricas que con una división típica, se premia la sencillez con mejor resultado sobre la complejidad por un resultado no mucho mejor.
