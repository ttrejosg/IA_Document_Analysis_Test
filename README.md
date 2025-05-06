# Proyecto de Pruebas de Modelos de IA

Este proyecto está diseñado para probar modelos de IA, específicamente `NousResearch/Meta-Llama-3.1-8B-Instruct` y `Qwen/Qwen2.5-7B-Instruct`, utilizando el conjunto de datos `naver-clova-ix/cord-v2`. El proyecto utiliza Python con `transformers` y `torch` como dependencias.

## Estructura del Proyecto

- `data/`: Contiene scripts relacionados con el conjunto de datos.
- `main.py`: Script principal para ejecutar el proyecto.
- `requirements.txt`: Lista las dependencias necesarias de Python.

## Configuración

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecutar el script principal:
   ```bash
   python main.py
   ```

## Conjunto de Datos

Se utiliza el conjunto de datos `naver-clova-ix/cord-v2`. La columna `image` se convierte a texto y la columna `ground_truth` se preprocesa para eliminar el campo `metadata`.

## Métricas de Evaluación

En este proyecto, se utilizan las siguientes métricas para evaluar la calidad de las respuestas generadas por los modelos:

### BLEU (Bilingual Evaluation Understudy)

La métrica BLEU mide la similitud entre el texto generado por el modelo y el texto de referencia. Se basa en la coincidencia de n-gramas y evalúa qué tan bien el modelo reproduce patrones similares al texto de referencia. En este proyecto, se utiliza la función `sentence_bleu` de la biblioteca `nltk` para calcular esta métrica.

- **Cómo funciona**: Compara n-gramas (secuencias de palabras) entre el texto generado y el texto de referencia.
- **Resultado**: Un puntaje entre 0 y 1, donde 1 indica una coincidencia perfecta.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

La métrica ROUGE evalúa la calidad del texto generado comparándolo con el texto de referencia. En este proyecto, se utiliza la función `rouge.get_scores` de la biblioteca `rouge` para calcular las variantes ROUGE-1, ROUGE-2 y ROUGE-L.

- **ROUGE-1**: Coincidencia de unigramas (palabras individuales).
- **ROUGE-2**: Coincidencia de bigramas (pares de palabras consecutivas).
- **ROUGE-L**: Coincidencia basada en la subsecuencia común más larga (LCS).
- **Resultado**: Proporciona precisión, recall y F1-score para cada variante.
