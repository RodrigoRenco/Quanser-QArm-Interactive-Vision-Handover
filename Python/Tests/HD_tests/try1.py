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
    return all(
        hand_landmarks.landmark[t].y < hand_landmarks.landmark[t - 2].y
        for t in tips
    )

# =========================
# GESTOS
# =========================

def es_palm_stop(hand_landmarks):
    if not es_mano_abierta(hand_landmarks):
        return False

    wrist = hand_landmarks.landmark[0]

    dedos_y = [hand_landmarks.landmark[t].y for t in [8,12,16,20]]
    vertical = all(d < wrist.y for d in dedos_y)

    dedos_z = [hand_landmarks.landmark[t].z for t in [8,12,16,20]]
    palm_z = hand_landmarks.landmark[9].z

    hacia_camara = (sum(dedos_z)/4) < palm_z - 0.02

    return vertical and hacia_camara


def es_offer(hand_landmarks):
    if not es_mano_abierta(hand_landmarks):
        return False

    dedos_y = [hand_landmarks.landmark[t].y for t in [8,12,16,20]]
    variacion_y = max(dedos_y) - min(dedos_y)
    horizontal = variacion_y < 0.12

    dedos_z = [hand_landmarks.landmark[t].z for t in [8,12,16,20]]
    palm_z = hand_landmarks.landmark[9].z

    no_frontal = abs((sum(dedos_z)/4) - palm_z) < 0.05

    return horizontal and no_frontal


# =========================
# DETECCIÓN SIMPLE DE OBJETO
# =========================

def hay_objeto(hand_landmarks):
    palma = hand_landmarks.landmark[9]

    dedos = [
        hand_landmarks.landmark[8],
        hand_landmarks.landmark[12],
        hand_landmarks.landmark[16],
        hand_landmarks.landmark[20],
    ]

    distancias = [
        ((d.x - palma.x)**2 + (d.y - palma.y)**2)**0.5
        for d in dedos
    ]

    promedio = sum(distancias) / len(distancias)

    return promedio > 0.18  # ajustar según pruebas


# =========================
# DETECTOR
# =========================

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

    gesto = "NONE"
    color = (255, 255, 255)
    hay_mano = False

    grasp_point = None
    objeto = False

    if results.multi_hand_landmarks:
        hay_mano = True

        for hand_landmarks in results.multi_hand_landmarks:

            gesto = "HAND"

            # =========================
            # DETECCIÓN DE GESTOS
            # =========================
            if es_offer(hand_landmarks):
                gesto = "OFFER"
                color = (0, 200, 255)

            elif es_palm_stop(hand_landmarks):
                gesto = "STOP"
                color = (0, 255, 255)

            # =========================
            # POSICIÓN MANO
            # =========================
            h, w, _ = frame.shape
            palm = hand_landmarks.landmark[9]

            cx = int(palm.x * w)
            cy = int(palm.y * h)

            # =========================
            # DETECCIÓN OBJETO
            # =========================
            objeto = hay_objeto(hand_landmarks)

            # =========================
            # GRASP POINT
            # =========================
            if gesto == "OFFER":
                grasp_point = (cx, cy - 40)

            elif gesto == "STOP":
                grasp_point = (cx + 40, cy)

            # =========================
            # VISUAL DEBUG
            # =========================
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

            if grasp_point:
                cv2.circle(frame, grasp_point, 10, (0, 0, 255), -1)

            if objeto:
                cv2.putText(frame, "OBJECT", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

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

    # =========================
    # OUTPUT VISUAL
    # =========================
    cv2.putText(frame, gesto_final, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gestos", frame)

    # =========================
    # OUTPUT PARA TU COMPAÑERO
    # =========================
    data = {
        "gesture": gesto_final,
        "grasp_point": grasp_point,
        "object_detected": objeto
    }

    print(data)  # puedes reemplazar esto por socket/serial

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()