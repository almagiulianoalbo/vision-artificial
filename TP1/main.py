import cv2
import mediapipe as mp
import numpy as np

# Inicializamos el módulo Holistic de MediaPipe.
# Holistic permite detectar cara, pose y manos.
# En este proyecto estamos usando sobre todo las manos.
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,  # confianza mínima para detectar
    min_tracking_confidence=0.5  # confianza mínima para seguir el movimiento
)

# Utilidad de MediaPipe para dibujar landmarks y conexiones.
mp_drawing = mp.solutions.drawing_utils

# Color rojo en formato BGR para OpenCV.
RED_COLOR = (0, 0, 255)

# Abrimos la cámara web.
capture = cv2.VideoCapture(0)

# Verificamos que la cámara se haya abierto correctamente.
if not capture.isOpened():
    print("Error: no se pudo abrir la camara")
    exit()


def calculate_distance(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos de la mano.
    Cada punto tiene coordenadas x e y normalizadas por MediaPipe.
    """
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def finger_is_extended(hand_landmarks, tip_idx, pip_idx, wrist_idx=0):
    """
    Determina si un dedo está extendido.

    Idea:
    - tip_idx: índice del punto de la punta del dedo
    - pip_idx: índice de una articulación intermedia del dedo
    - wrist_idx: índice de la muñeca

    Comparamos:
    - distancia de la punta del dedo a la muñeca
    - distancia de la articulación intermedia a la muñeca

    Si la punta está más lejos de la muñeca que la articulación,
    asumimos que el dedo está extendido.
    """
    tip = hand_landmarks.landmark[tip_idx]
    pip = hand_landmarks.landmark[pip_idx]
    wrist = hand_landmarks.landmark[wrist_idx]

    tip_dist = calculate_distance(tip, wrist)
    pip_dist = calculate_distance(pip, wrist)

    return tip_dist > pip_dist


def get_finger_states(hand_landmarks):
    """
    Devuelve un diccionario indicando si cada dedo está extendido o no.

    Para índice, medio, anular y meñique usamos la función finger_is_extended.
    Para el pulgar hacemos una aproximación similar comparando la punta
    y una articulación del pulgar respecto de la muñeca.
    """
    if hand_landmarks is None:
        return None

    index_ext = finger_is_extended(hand_landmarks, 8, 6)
    middle_ext = finger_is_extended(hand_landmarks, 12, 10)
    ring_ext = finger_is_extended(hand_landmarks, 16, 14)
    pinky_ext = finger_is_extended(hand_landmarks, 20, 18)

    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    thumb_ext = calculate_distance(thumb_tip, wrist) > calculate_distance(thumb_ip, wrist)

    return {
        "thumb": thumb_ext,
        "index": index_ext,
        "middle": middle_ext,
        "ring": ring_ext,
        "pinky": pinky_ext
    }


def hand_center(hand_landmarks):
    """
    Calcula el centro aproximado de una mano como el promedio de todos
    los landmarks en x e y.

    Esto sirve para comparar posiciones relativas entre manos:
    por ejemplo, si están cerca o si una está por encima de la otra.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    return np.mean(xs), np.mean(ys)


def draw_hand_landmarks(image, hand_landmarks):
    """
    Dibuja los landmarks de la mano y sus conexiones:
    - líneas blancas finas
    - puntos rojos pequeños

    Esto no afecta la detección.
    Solo sirve para visualizar qué está viendo la computadora.
    """
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)
    )


def get_landmark_xyz(hand_landmarks, idx):
    """
    Devuelve un landmark como vector (x, y, z).
    Acá usamos también z porque MediaPipe estima cierta profundidad relativa.
    """
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def palm_visibility_score(hand_landmarks):
    """
    Calcula un puntaje aproximado de cuán visible está la palma para la cámara.

    Idea:
    - tomamos tres puntos:
      muñeca (0), base del índice (5), base del meñique (17)
    - esos tres puntos definen aproximadamente el plano de la palma
    - con dos vectores sobre ese plano calculamos el vector normal
    - observamos la componente z de esa normal

    Si el valor es alto:
    - la palma está más 'de frente' a la cámara

    Si el valor es bajo:
    - la mano está más de canto

    Esto no detecta perfectamente "palma hacia arriba" física,
    pero sí detecta si la palma se ve bien para la cámara.
    """
    wrist = get_landmark_xyz(hand_landmarks, 0)
    index_mcp = get_landmark_xyz(hand_landmarks, 5)
    pinky_mcp = get_landmark_xyz(hand_landmarks, 17)

    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    if norm < 1e-6:
        return 0.0

    normal = normal / norm

    return abs(float(normal[2]))


"""SEÑA: AYUDA"""


def is_open_palm(hand_landmarks):
    """
    Consideramos mano abierta cuando:
    índice, medio, anular y meñique están extendidos.

    No usamos el pulgar como condición obligatoria porque su orientación
    puede variar mucho según la persona y el ángulo de la cámara.
    """
    states = get_finger_states(hand_landmarks)
    if states is None:
        return False

    return (
            states["index"] and
            states["middle"] and
            states["ring"] and
            states["pinky"]
    )


def is_palm_visible_for_help(hand_landmarks):
    """
    Para la seña AYUDA exigimos que la palma esté lo bastante visible.

    Si el score supera 0.35, consideramos que la palma está orientada
    suficientemente hacia la cámara.
    """
    score = palm_visibility_score(hand_landmarks)
    return score > 0.35


def is_fist(hand_landmarks):
    """
    Consideramos puño cuando índice, medio, anular y meñique NO están extendidos.
    """
    states = get_finger_states(hand_landmarks)
    if states is None:
        return False

    return (
            not states["index"] and
            not states["middle"] and
            not states["ring"] and
            not states["pinky"]
    )


def is_help_sign(left_hand, right_hand):
    """
    Detecta la seña AYUDA usando dos manos.

    Lógica:
    - una mano debe estar abierta y con la palma visible
    - la otra debe estar en puño
    - ambas manos deben estar relativamente cerca
    - el puño debe estar por encima o casi por encima de la palma

    Consideramos ambos casos:
    - izquierda abierta + derecha puño
    - derecha abierta + izquierda puño
    """
    if left_hand is None or right_hand is None:
        return False

    left_open_visible = is_open_palm(left_hand) and is_palm_visible_for_help(left_hand)
    right_open_visible = is_open_palm(right_hand) and is_palm_visible_for_help(right_hand)

    left_fist = is_fist(left_hand)
    right_fist = is_fist(right_hand)

    lx, ly = hand_center(left_hand)
    rx, ry = hand_center(right_hand)

    # Las manos deben estar cerca en horizontal y vertical.
    hands_close_x = abs(lx - rx) < 0.20
    hands_close_y = abs(ly - ry) < 0.24

    # Caso 1: izquierda abierta, derecha en puño.
    if left_open_visible and right_fist:
        fist_above = ry < ly + 0.08
        return hands_close_x and hands_close_y and fist_above

    # Caso 2: derecha abierta, izquierda en puño.
    if right_open_visible and left_fist:
        fist_above = ly < ry + 0.08
        return hands_close_x and hands_close_y and fist_above

    return False


"""SEÑA: DOLOR"""


def is_index_only(hand_landmarks):
    """
    Consideramos que la mano hace la forma de DOLOR cuando:
    - el índice está extendido
    - medio, anular y meñique están doblados
    """
    states = get_finger_states(hand_landmarks)
    if states is None:
        return False

    return (
            states["index"] and
            not states["middle"] and
            not states["ring"] and
            not states["pinky"]
    )


def is_pain_sign(left_hand, right_hand):
    """
    Detecta la seña DOLOR usando dos manos.

    Lógica:
    - ambas manos deben tener solo el índice extendido
    - las puntas de ambos índices deben estar muy cerca entre sí
    """
    if left_hand is None or right_hand is None:
        return False

    left_index_only = is_index_only(left_hand)
    right_index_only = is_index_only(right_hand)

    if not (left_index_only and right_index_only):
        return False

    left_index_tip = left_hand.landmark[8]
    right_index_tip = right_hand.landmark[8]

    index_distance = calculate_distance(left_index_tip, right_index_tip)

    # Umbral de cercanía entre los índices.
    return index_distance < 0.07


"""SEÑA: LLAMAR"""


def is_call_sign(hand_landmarks):
    """
    Detecta la seña LLAMAR con una sola mano.

    Lógica:
    - pulgar extendido
    - índice, medio y anular doblados
    - meñique extendido
    - además el pulgar y el meñique deben estar bastante separados

    Esto intenta capturar la forma de "teléfono".
    """
    states = get_finger_states(hand_landmarks)
    if states is None:
        return False

    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_pinky_distance = calculate_distance(thumb_tip, pinky_tip)

    return (
            states["thumb"] and
            not states["index"] and
            not states["middle"] and
            not states["ring"] and
            states["pinky"] and
            thumb_pinky_distance > 0.30
    )


while capture.isOpened():
    # Leemos un frame de la cámara.
    ret, frame = capture.read()
    if not ret:
        break

    # OpenCV trabaja en BGR, pero MediaPipe procesa en RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Marcamos la imagen como no escribible para optimizar el procesamiento.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Volvemos a BGR para mostrar la imagen con OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extraemos landmarks de mano izquierda y derecha.
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    # Dibujamos lo que MediaPipe detecta.
    if left_hand is not None:
        draw_hand_landmarks(image, left_hand)

    if right_hand is not None:
        draw_hand_landmarks(image, right_hand)

    # Determinamos si hay una o dos manos detectadas.
    two_hands = left_hand is not None and right_hand is not None
    one_hand = (left_hand is not None) ^ (right_hand is not None)

    help_detected = False
    pain_detected = False
    call_detected = False

    # Si hay dos manos, evaluamos AYUDA y DOLOR.
    if two_hands:
        help_detected = is_help_sign(left_hand, right_hand)
        pain_detected = is_pain_sign(left_hand, right_hand)

    # Si hay una sola mano, evaluamos LLAMAR.
    if one_hand:
        if left_hand is not None:
            call_detected = is_call_sign(left_hand)
        else:
            call_detected = is_call_sign(right_hand)

    # Mostramos en pantalla el gesto detectado.
    # Usamos prioridad:
    # 1) AYUDA
    # 2) DOLOR
    # 3) LLAMAR
    if help_detected:
        cv2.putText(image, "AYUDA detectada", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)

    elif pain_detected:
        cv2.putText(image, "DOLOR detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)

    elif call_detected:
        cv2.putText(image, "LLAMAR detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)

    # Mostramos la imagen final en una ventana.
    cv2.imshow("Image", image)

    # Si el usuario toca la tecla q, termina el programa.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos ventanas.
capture.release()
cv2.destroyAllWindows()
