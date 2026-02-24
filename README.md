# Algoritmo K-Nearest Neighbor (KNN)

## Introducción

El K-Nearest Neighbor (KNN) es uno de los algoritmos de clasificación más simples y efectivos. Es un modelo basado en instancias, lo que significa que no realiza ningún aprendizaje explícito durante la fase de entrenamiento, sino que almacena los ejemplos de entrenamiento y clasifica los nuevos ejemplos basándose en la proximidad a los ejemplos almacenados.

### ¿Cómo funciona?

KNN clasifica un nuevo punto de datos observando los `k` puntos de datos más cercanos en el conjunto de entrenamiento y asignándole la clase más común (mayoritaria) entre ellos. Este algoritmo es intuitivo porque se basa en la idea de "si se parece a lo que ya conozco, debería ser lo mismo".

Por ejemplo, si quisiéramos clasificar un tipo de flor, KNN buscaría las `k` flores más parecidas en nuestro conjunto de datos y vería a qué tipo de flor pertenecen esas `k` flores cercanas para asignarle la misma categoría.

KNN es muy útil cuando los datos tienen relaciones complejas que no son lineales, y es fácil de entender y de implementar.

## Explicación Matemática

El algoritmo KNN se basa en la distancia entre puntos de datos. Una de las distancias más comunes es la **distancia Euclidiana**.

### Distancia Euclidiana

La distancia Euclidiana entre dos puntos en un espacio `n`-dimensional es la longitud de la línea recta que los une, y se calcula de la siguiente forma:

$$
d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

Donde:
- `p` y `q` son los puntos en el espacio.
- `pi` y `qi` son las coordenadas de los puntos en la dimensión `i`.

### Clasificación con KNN

Para clasificar un nuevo punto:
1. Calcula la distancia entre el nuevo punto y todos los puntos en el conjunto de datos de entrenamiento.
2. Selecciona los `k` puntos más cercanos (es decir, aquellos con las distancias más pequeñas).
3. La clase más común entre estos `k` vecinos es la clase asignada al nuevo punto.

El valor de `k` es un hiperparámetro que se debe elegir cuidadosamente. Un `k` pequeño puede hacer que el modelo sea sensible al ruido, mientras que un `k` grande puede hacer que el modelo pierda detalles importantes.

## Ejemplo Práctico: Clasificación de Tipos de Frutas

Imaginemos que tenemos un conjunto de datos de frutas, donde cada fruta está descrita por su peso y su tamaño. Queremos clasificar una nueva fruta basándonos en estas características.

## Paso a Paso con Código

### Paso 1: Importar librerías y crear conjunto de datos

```python
# Importar las bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Crear el conjunto de datos original
data = {
    'Peso': [150, 170, 140, 130, 155, 180, 200, 120, 135, 145],
    'Tamaño': [7.0, 7.5, 6.5, 6.0, 7.2, 8.0, 8.5, 5.5, 6.2, 6.8],
    'Tipo': ['Manzana', 'Manzana', 'Manzana', 'Manzana', 'Manzana', 'Naranja', 'Naranja', 'Naranja', 'Naranja', 'Naranja']
}

# Crear un DataFrame con los datos originales
df = pd.DataFrame(data)

# Duplicar el conjunto de datos para obtener veinte elementos
np.random.seed(42)  # Para reproducibilidad
additional_data = {
    'Peso': np.random.choice(data['Peso'], size=10, replace=True) + np.random.randint(-10, 10, size=10),
    'Tamaño': np.random.choice(data['Tamaño'], size=10, replace=True) + np.random.uniform(-0.5, 0.5, size=10),
    'Tipo': np.random.choice(data['Tipo'], size=10, replace=True)
}

# Crear un DataFrame con los datos adicionales
df_additional = pd.DataFrame(additional_data)

# Combinar ambos DataFrames
df = pd.concat([df, df_additional], ignore_index=True)

# Visualizar el DataFrame combinado
df.head(20)
```

### Paso 2: Separar datos para entrenamiento y prueba

```python
# Separar las características (X) y la etiqueta (y)
X = df[['Peso', 'Tamaño']]
y = df['Tipo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Paso 3: Entrenamiento del modelo KNN

```python
# Crear y entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

### Paso 4: Predicción y evaluación del modelo

```python
# Importar las bibliotecas necesarias
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Hacer predicciones sobre el conjunto de prueba
y_pred = knn.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Ajustar zero_division para evitar warnings en el reporte de clasificación
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
```

### Paso 5: Visualización de los resultados

```python
# Visualización de los datos y la clasificación
plt.figure(figsize=(10, 6))

# Puntos de entrenamiento
plt.scatter(X_train['Peso'], X_train['Tamaño'],
            c=[{'Manzana': 'red', 'Naranja': 'orange'}[x] for x in y_train],
            label='Datos de Entrenamiento')

# Puntos de prueba
plt.scatter(X_test['Peso'], X_test['Tamaño'],
            c=[{'Manzana': 'red', 'Naranja': 'orange'}[x] for x in y_pred],
            marker='x', label='Datos de Prueba')

plt.xlabel('Peso')
plt.ylabel('Tamaño')
plt.title('Clasificación de Frutas con KNN')
plt.legend()
plt.show()
```

## Interpretación de los Resultados

### Gráfico de la Clasificación

En el gráfico, los puntos circulares representan las frutas del conjunto de entrenamiento, donde las manzanas están en rojo y las naranjas en naranja. Los puntos en forma de "x" representan las frutas del conjunto de prueba, que han sido clasificadas por el modelo.

### Explicación de los Resultados

El modelo KNN ha sido capaz de clasificar correctamente la mayoría de las frutas basándose en su proximidad a los puntos de entrenamiento. La precisión obtenida y la matriz de confusión nos muestran cuán bien funciona el modelo para este conjunto de datos. Un valor de `k=3` significa que el modelo toma en cuenta los 3 vecinos más cercanos para tomar la decisión final de clasificación.

## Ventajas y Desventajas del KNN

### Ventajas:
- **Simplicidad**: Fácil de entender e implementar
- **No paramétrico**: No hace suposiciones sobre la distribución de los datos
- **Adaptable**: Puede usarse tanto para clasificación como para regresión
- **Funciona bien con datos no lineales**: Puede capturar relaciones complejas

### Desventajas:
- **Computacionalmente costoso**: Debe calcular distancias a todos los puntos de entrenamiento
- **Sensible a la escala**: Las características con rangos más grandes pueden dominar
- **Sensible al ruido**: Puntos atípicos pueden afectar las predicciones
- **Requiere mucha memoria**: Debe almacenar todo el conjunto de entrenamiento

## Consideraciones Importantes

1. **Elección de k**: Debe ser un número impar para evitar empates en clasificación binaria
2. **Normalización**: Es recomendable normalizar las características cuando tienen diferentes escalas
3. **Maldición de la dimensionalidad**: El rendimiento puede deteriorarse con muchas características
4. **Validación cruzada**: Usar para encontrar el valor óptimo de k

## Cuándo Usar KNN

KNN es especialmente útil cuando:
- Los datos tienen relaciones complejas no lineales
- Se tiene un conjunto de datos pequeño a mediano
- La interpretabilidad es importante
- Se necesita una solución rápida de implementar
- Los datos están bien distribuidos y no tienen mucho ruido
