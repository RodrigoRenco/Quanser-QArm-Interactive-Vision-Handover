# pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =========================
# FUNCIONES BASE
# =========================

def es_mano_abierta(hand_landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            return False
    return True


# =========================
# GESTOS
# =========================

def es_puno(hand_landmarks):
    tips = [8, 12, 16, 20]
    return all(
        hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
        for t in tips
    )


def es_pulgar_arriba(hand_landmarks):
    pulgar_tip = hand_landmarks.landmark[4]
    pulgar_ip = hand_landmarks.landmark[3]

    pulgar_arriba = pulgar_tip.y < pulgar_ip.y

    otros_dedos = [8, 12, 16, 20]
    cerrados = all(
        hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
        for t in otros_dedos
    )

    return pulgar_arriba and cerrados


def es_palm_stop(hand_landmarks):
    if not es_mano_abierta(hand_landmarks):
        return False

    wrist = hand_landmarks.landmark[0]
    dedos_y = [
        hand_landmarks.landmark[8].y,
        hand_landmarks.landmark[12].y,
        hand_landmarks.landmark[16].y,
        hand_landmarks.landmark[20].y,
    ]

    # Dedos arriba de la muñeca → vertical
    vertical = all(d < wrist.y for d in dedos_y)

    # Palma hacia cámara → diferencia en Z (dedos más cerca)
    dedos_z = [hand_landmarks.landmark[t].z for t in [8,12,16,20]]
    palm_z = hand_landmarks.landmark[9].z

    hacia_camara = (sum(dedos_z)/4) < palm_z - 0.02

    return vertical and hacia_camara


def es_offer(hand_landmarks):
    if not es_mano_abierta(hand_landmarks):
        return False

    wrist = hand_landmarks.landmark[0]
    palm = hand_landmarks.landmark[9]

    # Mano NO vertical
    dedos_y = [
        hand_landmarks.landmark[8].y,
        hand_landmarks.landmark[12].y,
        hand_landmarks.landmark[16].y,
        hand_landmarks.landmark[20].y,
    ]

    variacion_y = max(dedos_y) - min(dedos_y)
    horizontal = variacion_y < 0.12

    # Palma NO mirando a la cámara (clave)
    dedos_z = [hand_landmarks.landmark[t].z for t in [8,12,16,20]]
    promedio_dedos_z = sum(dedos_z) / 4

    no_frontal = abs(promedio_dedos_z - palm.z) < 0.05

    return horizontal and no_frontal


def es_ok(hand_landmarks):
    pulgar = hand_landmarks.landmark[4]
    indice = hand_landmarks.landmark[8]

    distancia = ((pulgar.x - indice.x)**2 + (pulgar.y - indice.y)**2)**0.5
    return distancia < 0.05


# =========================
# DETECTOR
# =========================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# =========================
# DETECTOR
# =========================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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

    gesto = "NONE"
    color = (255, 255, 255)
    hay_mano = False

    if results.multi_hand_landmarks:
        hay_mano = True

        for hand_landmarks in results.multi_hand_landmarks:

            gesto = "HAND"

            # =========================
            # ORDEN CLAVE
            # =========================
            if es_puno(hand_landmarks):
                gesto = "FIST"
                color = (0, 0, 255)

            elif es_pulgar_arriba(hand_landmarks):
                gesto = "THUMB UP"
                color = (0, 255, 0)

            elif es_offer(hand_landmarks):
                gesto = "OFFER"
                color = (0, 200, 255)

            elif es_palm_stop(hand_landmarks):
                gesto = "STOP"
                color = (0, 255, 255)

            elif es_ok(hand_landmarks):
                gesto = "OK"
                color = (255, 0, 255)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # =========================
    # LÓGICA FINAL
    # =========================
    if not hay_mano:
        gesto_final = "NONE"
        historial.clear()
    else:
        if gesto == "HAND":
            gesto_final = "HAND"
        else:
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