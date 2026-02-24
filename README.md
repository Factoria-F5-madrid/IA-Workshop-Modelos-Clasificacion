# Evaluación de Modelos de Clasificación

## Introducción

Cuando entrenamos un modelo de clasificación, uno de los pasos más importantes es evaluar qué tan bien está funcionando. Para esto, existen varias métricas que nos ayudan a entender el rendimiento del modelo más allá de simplemente la precisión.

En este ejercicio aprenderemos sobre las principales métricas de evaluación para modelos de clasificación:
- Matriz de Confusión
- Precisión (Precision)
- Sensibilidad (Recall)
- Puntuación F1 (F1 Score)
- AUC-ROC

## Marco Teórico

### Matriz de Confusión

La matriz de confusión es una tabla que permite visualizar el rendimiento de un algoritmo de clasificación. Cada columna de la matriz representa las instancias predichas como una clase específica, mientras que cada fila representa las instancias de la clase real.

```
                    Clase Predicha
Clase Real      Positivo    Negativo
Positivo          TP          FN
Negativo          FP          TN
```

Donde:
- **TP (True Positive)**: Verdaderos Positivos - Casos correctamente clasificados como positivos
- **TN (True Negative)**: Verdaderos Negativos - Casos correctamente clasificados como negativos
- **FP (False Positive)**: Falsos Positivos - Casos incorrectamente clasificados como positivos (Error Tipo I)
- **FN (False Negative)**: Falsos Negativos - Casos incorrectamente clasificados como negativos (Error Tipo II)

### Precisión (Precision)

La precisión nos dice qué proporción de las predicciones positivas fueron correctas.

```
Precisión = TP / (TP + FP)
```

**Interpretación**: De todos los casos que el modelo predijo como positivos, ¿qué porcentaje realmente eran positivos?

### Sensibilidad (Recall) o Tasa de Verdaderos Positivos

La sensibilidad (o recall) mide la proporción de positivos reales que fueron identificados correctamente por el modelo.

```
Sensibilidad (Recall) = TP / (TP + FN)
```

**Interpretación**: De todos los casos que realmente son positivos, ¿qué porcentaje fue correctamente identificado por el modelo?

### Puntuación F1 (F1 Score)

La puntuación F1 es la media armónica entre la precisión y la sensibilidad. Es útil cuando necesitamos un balance entre las dos métricas.

```
F1 = 2 × (Precisión × Recall) / (Precisión + Recall)
```

**Interpretación**: Combina precisión y recall en una sola métrica, útil cuando hay desequilibrio entre las clases.

### Curva AUC-ROC

La curva ROC (Receiver Operating Characteristic) es un gráfico que muestra la tasa de verdaderos positivos frente a la tasa de falsos positivos a diferentes umbrales de clasificación. El AUC (Area Under the Curve) mide el área bajo la curva ROC y da una idea general del rendimiento del modelo.

```
AUC-ROC = ∫ ROC(x) dx
```

**Interpretación**: 
- AUC = 1.0: Modelo perfecto
- AUC = 0.5: Modelo sin capacidad de discriminación (equivale a adivinar al azar)
- AUC > 0.8: Buen modelo
- AUC < 0.6: Modelo pobre

## Ejemplo Práctico: Clasificación de Clientes en un Banco

Imaginemos que trabajamos en un banco y queremos construir un modelo que prediga si un cliente hará un depósito a largo plazo (sí/no) basado en características como el saldo de la cuenta, si tiene tarjeta de crédito, y si es un cliente activo.

### Paso 1: Creación del Conjunto de Datos e Importación de Bibliotecas

**Código a implementar:**

```python
# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Ampliar el conjunto de datos simulado
data = {
    'Saldo': [
        20000, 5000, 15000, 8000, 12000, 7000, 22000, 18000, 13000, 4000,  # Originales
        25000, 3000, 17000, 9000, 11000, 6000, 24000, 19000, 14000, 4500,
        21000, 10000, 16000, 8500, 12500, 7500, 23000, 18500, 13500, 4200,
        22000, 5500, 15500, 8200, 12200, 6800, 22500, 17500, 13200, 4300
    ],
    'Tiene_Tarjeta': [
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0,  # Originales
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0
    ],
    'Es_Activo': [
        1, 1, 0, 0, 1, 1, 1, 0, 0, 1,  # Originales
        1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
        0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
        1, 1, 0, 0, 1, 1, 1, 0, 0, 1
    ],
    'Deposito': [
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0,  # Originales
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
        1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 1, 0, 0
    ]
}

# Crear un DataFrame con los datos ampliados
df = pd.DataFrame(data)

# Visualizar el DataFrame
df.head(10)
```

**Explicación del Paso 1:**
- Importamos las librerías necesarias para el análisis de datos y machine learning
- Creamos un dataset simulado de un banco con las siguientes características:
  - `Saldo`: Saldo en la cuenta del cliente
  - `Tiene_Tarjeta`: Si el cliente tiene tarjeta de crédito (1=Sí, 0=No)
  - `Es_Activo`: Si el cliente es activo en el banco (1=Sí, 0=No)
  - `Deposito`: Variable objetivo - si hará un depósito (1=Sí, 0=No)

### Paso 2: Separación de Datos para Entrenamiento y Prueba

**Código a implementar:**

```python
# Separar las características (X) y la etiqueta (y)
X = df[['Saldo', 'Tiene_Tarjeta', 'Es_Activo']]
y = df['Deposito']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Explicación del Paso 2:**
- Separamos las características (features) de la variable objetivo (target)
- `X` contiene las 3 características que usaremos para predecir
- `y` contiene la variable que queremos predecir (si hará depósito o no)
- Dividimos los datos: 70% para entrenamiento, 30% para prueba
- `random_state=42` asegura que la división sea reproducible

### Paso 3: Entrenamiento del Modelo de Clasificación

**Código a implementar:**

```python
# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)
```

**Explicación del Paso 3:**
- Creamos un modelo de regresión logística
- La regresión logística es ideal para problemas de clasificación binaria
- `fit()` entrena el modelo usando los datos de entrenamiento
- El modelo aprende la relación entre las características y la decisión de hacer depósito

### Paso 4: Predicción y Cálculo de Métricas de Evaluación

**Código a implementar:**

```python
from sklearn.metrics import classification_report

# Hacer predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)
y_pred_prob = modelo.predict_proba(X_test)[:, 1]

# Calcular las métricas de evaluación
matriz_confusion = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Intentar calcular AUC solo si hay más de una clase en y_test
if len(set(y_test)) > 1:
    auc = roc_auc_score(y_test, y_pred_prob)
else:
    auc = "No se puede calcular AUC-ROC con una sola clase presente en y_test"

# Mostrar las métricas
print("Matriz de Confusión:\n", matriz_confusion)
print("Precisión:", precision)
print("Sensibilidad (Recall):", recall)
print("F1 Score:", f1)
print("AUC-ROC:", auc)
```

**Explicación del Paso 4:**
- `predict()`: Hace predicciones binarias (0 o 1)
- `predict_proba()`: Calcula las probabilidades de cada clase
- Calculamos todas las métricas de evaluación:
  - **Matriz de Confusión**: Muestra los TP, TN, FP, FN
  - **Precisión**: % de predicciones positivas que fueron correctas
  - **Recall (Sensibilidad)**: % de casos positivos reales que fueron detectados
  - **F1 Score**: Combina precisión y recall
  - **AUC-ROC**: Capacidad del modelo para discriminar entre clases

### Paso 5: Visualización de la Curva ROC

**Código a implementar:**

```python
# Generar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="Modelo (AUC = {:.2f})".format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
```

**Explicación del Paso 5:**
- `roc_curve()`: Calcula los puntos para graficar la curva ROC
- `fpr`: Tasa de Falsos Positivos (False Positive Rate)
- `tpr`: Tasa de Verdaderos Positivos (True Positive Rate)
- La línea diagonal representa un modelo sin capacidad de discriminación
- Mientras más alejada esté la curva de la diagonal, mejor es el modelo

## Interpretación de Resultados

### Análisis de la Matriz de Confusión

La matriz de confusión nos muestra cómo se distribuyen las predicciones correctas e incorrectas del modelo. Idealmente, queremos que los valores en la diagonal (TP y TN) sean altos, lo que indicaría un alto número de predicciones correctas.

### Precisión

La precisión nos dice qué porcentaje de las predicciones positivas del modelo son correctas. En nuestro caso, la precisión nos dice cuántos de los clientes que el modelo predijo que harían un depósito realmente lo hicieron.

### Sensibilidad (Recall)

La sensibilidad o recall nos muestra el porcentaje de clientes que realmente hicieron un depósito y fueron correctamente identificados por el modelo.

### F1 Score

El F1 Score es una métrica combinada que toma en cuenta tanto la precisión como la sensibilidad. Es útil cuando necesitamos un balance entre ambas.

### AUC-ROC

El AUC-ROC nos da una idea de cuán bien el modelo distingue entre las clases. Un valor más cercano a 1 indica que el modelo tiene un buen desempeño.

### Explicación del Gráfico ROC

La curva ROC muestra la relación entre la tasa de verdaderos positivos (sensibilidad) y la tasa de falsos positivos a medida que se varía el umbral de decisión.

- **Tasa de Verdaderos Positivos (TPR)**: Es la misma que la sensibilidad. Nos indica la proporción de positivos correctamente clasificados.
- **Tasa de Falsos Positivos (FPR)**: Indica la proporción de negativos que fueron incorrectamente clasificados como positivos.

Un buen modelo se curva hacia la esquina superior izquierda, lo que significa que tiene una alta tasa de verdaderos positivos y una baja tasa de falsos positivos. El AUC, que mide el área bajo la curva, es una métrica que resume la calidad del modelo.

## Conclusiones

1. **Múltiples Métricas**: Es importante evaluar modelos de clasificación usando múltiples métricas, no solo la precisión general.

2. **Contexto del Problema**: La elección de qué métrica optimizar depende del contexto:
   - Si los falsos positivos son costosos → Optimizar **Precisión**
   - Si los falsos negativos son costosos → Optimizar **Recall**
   - Para un balance general → Usar **F1 Score**

3. **Curva ROC**: Proporciona una vista completa del rendimiento del modelo a través de diferentes umbrales.

4. **AUC-ROC**: Es una excelente métrica resumen para comparar diferentes modelos.

## Ejercicio Práctico

Utiliza este README como guía para implementar cada paso en tu notebook de Jupyter:

1. Copia el código de cada paso en celdas separadas
2. Ejecuta cada celda secuencialmente
3. Analiza los resultados obtenidos
4. Modifica los datos o parámetros para ver cómo cambian las métricas
5. Experimenta con diferentes algoritmos de clasificación

¡Practica con diferentes conjuntos de datos para dominar la evaluación de modelos de clasificación!
