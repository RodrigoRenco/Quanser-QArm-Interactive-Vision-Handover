# pip install mediapipe==0.10.13 opencv-python
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
    return True


def es_ok(hand_landmarks):
    pulgar = hand_landmarks.landmark[4]
    indice = hand_landmarks.landmark[8]

    distancia = ((pulgar.x - indice.x)**2 + (pulgar.y - indice.y)**2)**0.5
    return distancia < 0.05


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


# =========================
# DETECTOR
# =========================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesto = "NINGUNO"
    color = (255, 255, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # 👇 IMPORTANTE: calcular dedos primero
            dedos = contar_dedos(hand_landmarks)

            # 👇 ORDEN CORRECTO
            if es_puno(hand_landmarks):
                gesto = "PUÑO"
                color = (0, 0, 255)

            elif es_ok(hand_landmarks):
                gesto = "OK"
                color = (255, 0, 0)

            elif es_palma(hand_landmarks):
                gesto = "PALMA"
                color = (0, 255, 0)

            elif dedos == 1:
                gesto = "1 DEDO"
                color = (0, 255, 255)

            elif dedos == 2:
                gesto = "2 DEDOS"
                color = (255, 0, 255)

            elif dedos == 3:
                gesto = "3 DEDOS"
                color = (255, 255, 0)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, gesto, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gestos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
