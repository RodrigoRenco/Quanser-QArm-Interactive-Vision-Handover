# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# BASE

def get_hand_scale(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    return max(xs) - min(xs), max(ys) - min(ys)

# BASE GESTURES

def es_puno(hand_landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            return False
    return True


def dedos_cerrados_excluyendo(hand_landmarks, excluir_indices):
    tips = [12, 16, 20]
    pips = [10, 14, 18]

    for tip, pip in zip(tips, pips):
        if tip not in excluir_indices:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                return False
    return True


# DIRECTIONAL INDEX POINTING DETECTION

def indice_apunta_en_rango(hand_landmarks, angulo_min, angulo_max):

    tip = hand_landmarks.landmark[8]
    pip = hand_landmarks.landmark[6]
    mcp = hand_landmarks.landmark[5]

    dx = tip.x - mcp.x
    dy = -(tip.y - mcp.y)

    longitud = math.sqrt(dx * dx + dy * dy)

    hand_width, hand_height = get_hand_scale(hand_landmarks)

    if longitud < hand_width * 0.25:
        return False

    if tip.y >= pip.y:
        return False

    if not dedos_cerrados_excluyendo(hand_landmarks, [8]):
        return False

    angulo = math.degrees(math.atan2(dy, dx))
    if angulo < 0:
        angulo += 360

    return angulo_min <= angulo <= angulo_max


# UPGRADED STOP GESTURE

def es_palm_stop(hand_landmarks):

    dedos = [8, 12, 16, 20]

    extendidos = 0
    for tip in dedos:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extendidos += 1

    if extendidos < 4:
        return False

    wrist = hand_landmarks.landmark[0]

    hand_width, hand_height = get_hand_scale(hand_landmarks)

    xs = [hand_landmarks.landmark[t].x for t in dedos]
    ys = [hand_landmarks.landmark[t].y for t in dedos]

    apertura = max(xs) - min(xs)

    # MORE ROBUST TO DISTANCE
    if apertura < hand_width * 0.45:
        return False

    if not all(y < wrist.y for y in ys):
        return False

    altura = max(ys) - min(ys)

    if altura > hand_height * 0.5:
        return False

    return True


# DIRECTIONAL DETECTION WITH HAND

def es_apuntar_derecha(hand_landmarks, tipo_mano):

    # INDEX
    if tipo_mano == "Right":
        index_ok = (
            indice_apunta_en_rango(hand_landmarks, 0, 60) or
            indice_apunta_en_rango(hand_landmarks, 330, 360)
        )
    else:
        index_ok = indice_apunta_en_rango(hand_landmarks, 120, 240)

    if index_ok:
        return True

    # THUMB
    hand_width, _ = get_hand_scale(hand_landmarks)
    thr = hand_width * 0.15

    p_tip = hand_landmarks.landmark[4]
    p_ip = hand_landmarks.landmark[3]

    if tipo_mano == "Right":
        ok = (p_tip.x - p_ip.x) > thr
    else:
        ok = (p_ip.x - p_tip.x) > thr

    if ok and dedos_cerrados_excluyendo(hand_landmarks, [4]):
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y:
            return True

    return False


def es_apuntar_izquierda(hand_landmarks, tipo_mano):

    if tipo_mano == "Right":
        index_ok = indice_apunta_en_rango(hand_landmarks, 120, 240)
    else:
        index_ok = (
            indice_apunta_en_rango(hand_landmarks, 0, 60) or
            indice_apunta_en_rango(hand_landmarks, 330, 360)
        )

    if index_ok:
        return True

    # THUMB
    hand_width, _ = get_hand_scale(hand_landmarks)
    thr = hand_width * 0.15

    p_tip = hand_landmarks.landmark[4]
    p_ip = hand_landmarks.landmark[3]

    if tipo_mano == "Right":
        ok = (p_ip.x - p_tip.x) > thr
    else:
        ok = (p_tip.x - p_ip.x) > thr

    if ok and dedos_cerrados_excluyendo(hand_landmarks, [4]):
        if hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y:
            return True

    return False


# DETECTOR

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

historial = []
HIST_SIZE = 5

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesto = "HAND"
    color = (255, 255, 255)

    hay_mano = False

    if results.multi_hand_landmarks:

        hay_mano = True

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            tipo_mano = handedness.classification[0].label  # Left / Right

            gesto = "HAND"

            if es_palm_stop(hand_landmarks):
                gesto = "STOP"
                color = (0, 255, 255)

            elif es_apuntar_derecha(hand_landmarks, tipo_mano):
                gesto = "RIGHT"
                color = (255, 0, 0)

            elif es_apuntar_izquierda(hand_landmarks, tipo_mano):
                gesto = "LEFT"
                color = (255, 0, 0)

            elif es_puno(hand_landmarks):
                gesto = "HAND"

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # TEMPORAL FILTER

    if not hay_mano:
        historial.clear()
        gesto_final = "NONE"

    else:
        historial.append(gesto)
        if len(historial) > HIST_SIZE:
            historial.pop(0)

        gesto_final = max(set(historial), key=historial.count)

    cv2.putText(
        frame,
        gesto_final,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Gestos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
