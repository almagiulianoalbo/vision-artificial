# Visión Artificial — Trabajos Prácticos

Repositorio con los trabajos prácticos de la materia **Procesamiento de Imágenes y Señales / Visión Artificial**.

El objetivo de este repositorio es documentar el desarrollo de distintos proyectos de visión artificial utilizando **Python, OpenCV y MediaPipe**, aplicados al análisis de imágenes en tiempo real.

Cada carpeta corresponde a un **trabajo práctico independiente** desarrollado durante la cursada.

---

# Estructura del repositorio

```
vision-artificial/
│
├── TP1/
│   └── main.py
├── TP2...
├── TP3...
└── TP4...
```

Cada TP incluye:

* código fuente
* explicación del enfoque utilizado
* posibles mejoras o limitaciones

---

# TP1 — Reconocimiento de gestos de Lengua de Señas

## Objetivo

El objetivo del primer trabajo práctico fue desarrollar un **sistema simple de reconocimiento de gestos** utilizando visión artificial en tiempo real.

El sistema detecta ciertas señas básicas de comunicación que pueden ser útiles para una persona con discapacidad auditiva en situaciones de emergencia o necesidad inmediata.

Las señas reconocidas son:

* **AYUDA**
* **DOLOR**
* **LLAMAR**

---

# Tecnologías utilizadas

* **Python**
* **OpenCV**
* **MediaPipe (Holistic)**
* **NumPy**

MediaPipe se utiliza para detectar **landmarks de la mano en tiempo real**.

Cada mano es modelada con **21 puntos clave (landmarks)** que representan articulaciones y extremos de los dedos.

A partir de estos puntos se construyen **reglas geométricas** para identificar cada gesto.

---

# Funcionamiento general

El sistema sigue los siguientes pasos:

1. Captura de video desde la webcam.
2. Detección de manos mediante **MediaPipe Holistic**.
3. Extracción de los **landmarks de cada mano**.
4. Cálculo de **distancias y relaciones espaciales entre puntos**.
5. Aplicación de **reglas geométricas** para clasificar el gesto.
6. Visualización del resultado en tiempo real.

Además se dibujan los **landmarks y las conexiones de la mano** para visualizar qué está detectando el modelo.

---

# Detección de dedos extendidos

Para determinar si un dedo está extendido se compara:

* distancia entre la **punta del dedo y la muñeca**
* distancia entre una **articulación intermedia y la muñeca**

Si la punta está más lejos que la articulación, el dedo se considera **extendido**.

Este criterio permite determinar la postura de cada dedo:

```
pulgar
índice
medio
anular
meñique
```

---

# Lógica de reconocimiento de gestos

## AYUDA

Condiciones:

* dos manos detectadas
* una mano **abierta**
* la otra mano en **puño**
* ambas manos **cerca entre sí**
* el puño aproximadamente **sobre la palma**

Además se calcula un **puntaje de visibilidad de la palma** utilizando el plano definido por tres landmarks de la mano. Esto permite estimar si la palma está orientada hacia la cámara.

---

## DOLOR

Condiciones:

* dos manos detectadas
* **solo el dedo índice extendido en ambas manos**
* las **puntas de los índices muy cercanas**

Esto se detecta midiendo la distancia entre los landmarks correspondientes a las puntas de ambos índices.

---

## LLAMAR

Condiciones:

* **una sola mano visible**
* pulgar extendido
* meñique extendido
* índice, medio y anular doblados
* distancia grande entre pulgar y meñique

Esto intenta capturar la forma de **"teléfono"** característica de esta seña.

---

# Visualización

El programa muestra:

* los **landmarks de la mano**
* las **conexiones entre articulaciones**
* el **gesto detectado** en pantalla

Esto permite verificar visualmente qué está detectando el sistema.

---

# Ejecución

Instalar dependencias:

```bash
pip install opencv-python mediapipe numpy
```

Ejecutar el script:

```bash
python main.py
```

Controles:

```
q → salir del programa
```

---

# Limitaciones

Este sistema utiliza **reglas geométricas simples**, por lo que:

* depende de la iluminación
* depende del ángulo de la mano respecto a la cámara
* no reconoce todas las variaciones posibles de una seña
* puede fallar si las manos se superponen

Sin embargo, permite demostrar el uso de **visión artificial en tiempo real para reconocer gestos manuales**.

---

# Posibles mejoras

* entrenar un modelo de **machine learning para clasificación de gestos**
* usar **redes neuronales** sobre secuencias de landmarks
* agregar más señas
* mejorar la robustez frente a variaciones de orientación

---

# Autor

Trabajo realizado para la materia **Procesamiento de Imágenes y Señales / Visión Artificial**

GRUPO 2: Giuliano Albo Alma, Valdez Lourdes, Zabalett Angelina

Ingeniería Biomédica.

