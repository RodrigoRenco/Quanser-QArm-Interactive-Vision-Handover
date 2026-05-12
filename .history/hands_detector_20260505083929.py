# pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =========================
# FUNCIONES DE GESTOS
# =========================

def es_puno(hand_landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            return False
    return True


def es_palma(hand_landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            return False

    # Pulgar también abierto
    return hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x


def es_ok(hand_landmarks):
    pulgar = hand_landmarks.landmark[4]
    indice = hand_landmarks.landmark[8]

    distancia = ((pulgar.x - indice.x)**2 + (pulgar.y - indice.y)**2)**0.5
    return distancia < 0.05


def es_pulgar_arriba(hand_landmarks):
    pulgar_tip = hand_landmarks.landmark[4]
    pulgar_ip = hand_landmarks.landmark[3]

    pulgar_arriba = pulgar_tip.y < pulgar_ip.y

    otros_dedos = [8, 12, 16, 20]
    cerrados = True
    for tip in otros_dedos:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            cerrados = False

    return pulgar_arriba and cerrados


def es_apuntar_derecha(hand_landmarks):
    indice_tip = hand_landmarks.landmark[8]
    indice_pip = hand_landmarks.landmark[6]

    extendido = indice_tip.x > indice_pip.x

    otros = [12, 16, 20]
    cerrados = all(
        hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
        for t in otros
    )

    return extendido and cerrados


def es_apuntar_izquierda(hand_landmarks):
    indice_tip = hand_landmarks.landmark[8]
    indice_pip = hand_landmarks.landmark[6]

    extendido = indice_tip.x < indice_pip.x

    otros = [12, 16, 20]
    cerrados = all(
        hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
        for t in otros
    )

    return extendido and cerrados


def contar_dedos(hand_landmarks):
    dedos = 0

    tips = [8, 12, 16, 20]
    pip = [6, 10, 14, 18]

    for t, p in zip(tips, pip):
        if hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y:
            dedos += 1

    # Pulgar
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        dedos += 1

    return dedos

def es_palma_arriba(hand_landmarks):
    # Primero: debe ser palma abierta
    if not es_palma(hand_landmarks):
        return False

    wrist = hand_landmarks.landmark[0]   # muñeca
    palm = hand_landmarks.landmark[9]    # centro de la mano

    # La palma "arriba" suele tener la muñeca más abajo (en imagen: mayor y)
    if wrist.y < palm.y:
        return False

    # Opcional: verificar que los dedos estén bastante alineados (horizontal)
    dedos_y = [
        hand_landmarks.landmark[8].y,
        hand_landmarks.landmark[12].y,
        hand_landmarks.landmark[16].y,
        hand_landmarks.landmark[20].y,
    ]

    variacion = max(dedos_y) - min(dedos_y)

    return variacion < 0.1  # umbral ajustable


# =========================
# DETECTOR
# =========================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# 🔥 Suavizado temporal
historial = []
HIST_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo (más natural)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesto = "NONE"
    color = (255, 255, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            dedos = contar_dedos(hand_landmarks)

            # =========================
            # ORDEN IMPORTANTE
            # =========================
            if es_puno(hand_landmarks):
                gesto = "FIST"
                color = (0, 0, 255)

            elif es_pulgar_arriba(hand_landmarks):
                gesto = "THUMB UP"
                color = (0, 255, 0)

            #elif es_apuntar_derecha(hand_landmarks):
            #    gesto = "RIGHT"
            #    color = (255, 0, 0)
                
            #elif es_apuntar_izquierda(hand_landmarks):
            #    gesto = "LEFT"
            #    color = (255, 0, 0)
                
            elif es_palma_arriba(hand_landmarks):
                gesto = "OFFER"
                color = (0, 200, 255)

            elif es_ok(hand_landmarks):
                gesto = "OK"
                color = (255, 0, 255)

            elif es_palma(hand_landmarks):
                gesto = "PALM"
                color = (0, 255, 255)

            elif dedos == 1:
                gesto = "1 FINGER"
                color = (255, 255, 0)

            elif dedos == 2:
                gesto = "2 FINGERS"
                color = (255, 255, 0)
                
            elif dedos == 3:
                gesto = "3 FINGERS"
                color = (255, 255, 0)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # =========================
    # SUAVIZADO
    # =========================
    historial.append(gesto)
    if len(historial) > HIST_SIZE:
        historial.pop(0)

    gesto_final = max(set(historial), key=historial.count)

    cv2.putText(frame, gesto_final, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gestos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()