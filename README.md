# Árboles de Decisión - Guía Completa para el Ejercicio

## Introducción Teórica

Los árboles de decisión son un tipo de algoritmo de Machine Learning que se utilizan para tomar decisiones basadas en una serie de preguntas y respuestas. Imagina que estás jugando un juego de "20 preguntas" en el que tienes que adivinar un objeto. Comienzas haciendo preguntas generales, como "¿Es un animal?" y luego haces preguntas más específicas dependiendo de la respuesta, como "¿Tiene cuatro patas?" o "¿Puede volar?". Al final, llegas a una conclusión basada en las respuestas obtenidas.

Este proceso de hacer preguntas y tomar decisiones en cada paso es muy similar a cómo funcionan los árboles de decisión. Los árboles de decisión son visualmente como un diagrama de flujo donde cada "nodo" representa una pregunta (o condición), y las "ramas" representan las posibles respuestas o decisiones que puedes tomar en función de esa pregunta.

## ¿Por qué son interesantes?

1. **Facilidad de interpretación**: Los árboles de decisión son fáciles de entender y explicar. Cada decisión tomada en el árbol es claramente visible, lo que hace que el modelo sea transparente.

2. **Visualización clara**: Puedes ver y entender cómo el modelo llega a una conclusión al seguir las ramas del árbol, lo que es muy útil para explicar el razonamiento detrás de una predicción.

3. **Flexibilidad**: Los árboles de decisión pueden ser usados para problemas tanto de clasificación como de regresión, lo que los hace muy versátiles.

## Matemáticas Detrás de los Árboles de Decisión

Para entender cómo los árboles de decisión toman decisiones, necesitamos hablar de dos conceptos importantes: **entropía** y **ganancia de información**.

### Entropía

La **entropía** es una medida de la incertidumbre o impureza en un conjunto de datos. En otras palabras, nos dice qué tan mezclados están los datos en términos de sus etiquetas. La fórmula matemática para la entropía es:

$$H(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$

Donde:
- $S$ es el conjunto de datos
- $p_i$ es la proporción de elementos en la clase $i$

Si todos los elementos en $S$ pertenecen a la misma clase, la entropía es 0 (no hay incertidumbre). Si las clases están perfectamente balanceadas, la entropía es 1 (máxima incertidumbre).

### Ganancia de Información

La **ganancia de información** es la reducción en la entropía que resulta al dividir un conjunto de datos en subconjuntos basados en una característica particular. La ganancia de información se calcula como:

$$IG(S, A) = H(S) - \sum_{v \in \text{Valores}(A)} \frac{|S_v|}{|S|} H(S_v)$$

Donde:
- $A$ es el atributo que estamos utilizando para dividir el conjunto
- $S_v$ es el subconjunto de $S$ donde el atributo $A$ toma el valor $v$

La característica con la mayor ganancia de información se selecciona para dividir el nodo en el árbol de decisión.

## Ejemplo Práctico: Decisión de Adopción de Mascotas

Imagina que estás trabajando en un refugio de animales y quieres predecir si una persona adoptará un perro basado en algunas preguntas simples. Tienes los siguientes datos:

- ¿La persona tiene un jardín?
- ¿La persona tiene hijos?
- ¿La persona es alérgica a los animales?

Estos factores se pueden utilizar para construir un árbol de decisión que prediga si la persona adoptará un perro o no.

## Paso a Paso para Realizar el Ejercicio

### Paso 1: Importa las librerías

Necesitarás importar las siguientes librerías:
- `pandas` para el manejo de datos
- `DecisionTreeClassifier` y `plot_tree` de `sklearn.tree` para crear y visualizar el árbol
- `matplotlib.pyplot` para mostrar las gráficas

**Código para la celda:**
```python
# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```

### Paso 2: Cargamos el dataset ficticio

En este paso crearás el conjunto de datos, lo convertirás a DataFrame y visualizarás los primeros registros.

**Código para la celda:**
```python
# Crear el conjunto de datos
data = {
    'Tiene_jardin': [1, 1, 0, 0, 1, 0, 0, 1],
    'Tiene_hijos': [0, 1, 1, 0, 0, 1, 0, 1],
    'Es_alergico': [0, 0, 0, 1, 1, 1, 0, 0],
    'Adopta_perro': [1, 1, 1, 0, 1, 0, 0, 1]
}

# Crear un DataFrame con los datos
df = pd.DataFrame(data)
df.head()
```

### Paso 3: Separa las variables independientes de la dependiente en X e Y

Divide los datos en características (X) y variable objetivo (y).

**Código para la celda:**
```python
# Separar las características (X) y la etiqueta (y)
X = df[['Tiene_jardin', 'Tiene_hijos', 'Es_alergico']]
y = df['Adopta_perro']
```

### Paso 4: Entrena el modelo

Crea y entrena el modelo de árbol de decisión.

**Código para la celda:**
```python
# Crear y entrenar el modelo de árbol de decisión
modelo = DecisionTreeClassifier(criterion='entropy', random_state=0)
modelo.fit(X, y)
```

### Paso 5: Visualiza el árbol de decisión

Crea la visualización del árbol entrenado.

**Código para la celda:**
```python
# Visualizar el árbol de decisión con ajustes en el tamaño del gráfico y disposición
plt.figure(figsize=(16,10))  # Aumentar el tamaño del gráfico
plot_tree(modelo, feature_names=X.columns, class_names=['No Adopta', 'Adopta'], filled=True, rounded=True, fontsize=12)
plt.show()
```

## Interpretación del Árbol de Decisión

Una vez generado el árbol, podrás interpretar los siguientes elementos:

### Nodo Raíz
- Realiza la primera división basada en la característica que más reduce la entropía
- Muestra la pregunta más importante para la clasificación

### Entropía en cada nodo
- **Entropía alta**: Mayor incertidumbre en los datos
- **Entropía baja**: Mayor certeza en la clasificación
- **Entropía = 0**: Todos los elementos pertenecen a una sola clase

### Valor (value)
El `value` mostrado en cada nodo representa la distribución de las clases:
- `value = [3, 5]` significa 3 personas que no adoptan y 5 que sí adoptan
- Ayuda a entender la proporción de clases en cada rama

### Ramas del Árbol
- **Rama Izquierda**: Generalmente representa "Sí" o valor 1
- **Rama Derecha**: Generalmente representa "No" o valor 0

## Análisis de Resultados Esperados

Según el ejemplo de adopción de mascotas, esperarías encontrar:

1. **Característica más importante**: "¿Tiene jardín?" - Esta será la primera división
2. **Segunda característica**: "¿Tiene hijos?" - Para personas sin jardín
3. **Patrones identificados**:
   - Personas con jardín: Alta probabilidad de adopción
   - Personas sin jardín pero con hijos: Probabilidad moderada-alta
   - Personas sin jardín y sin hijos: Baja probabilidad de adopción

## Conclusiones del Ejercicio

Al completar este ejercicio, habrás aprendido:

1. **Cómo crear un árbol de decisión** usando scikit-learn
2. **Interpretar la entropía** y su significado en la toma de decisiones
3. **Entender la ganancia de información** y cómo se seleccionan las características
4. **Visualizar y analizar** un modelo de Machine Learning de forma intuitiva
5. **Aplicar el conocimiento** a un problema real de clasificación

El árbol de decisión te revelará que la característica más determinante para predecir la adopción es si la persona tiene jardín, seguida por si tiene hijos. La entropía en cada nodo muestra cuán ciertas son las decisiones, y el valor proporciona la distribución clara de los datos en cada rama.

El árbol de decisión te revelará que la característica más determinante para predecir la adopción es si la persona tiene jardín, seguida por si tiene hijos. La entropía en cada nodo muestra cuán ciertas son las decisiones, y el valor proporciona la distribución clara de los datos en cada rama.

## Conceptos Clave para Recordar

- **Entropía**: Medida de incertidumbre (0 = certeza total, 1 = máxima incertidumbre)
- **Ganancia de Información**: Reducción de entropía al hacer una división
- **Nodo**: Punto de decisión en el árbol
- **Hoja**: Nodo final que contiene la predicción
- **Criterio de división**: Se basa en maximizar la ganancia de información