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

    palm = hand_landmarks.landmark[9]

    dedos_y = [hand_landmarks.landmark[t].y for t in [8,12,16,20]]
    variacion_y = max(dedos_y) - min(dedos_y)
    horizontal = variacion_y < 0.12

    dedos_z = [hand_landmarks.landmark[t].z for t in [8,12,16,20]]
    promedio_dedos_z = sum(dedos_z) / 4

    no_frontal = abs(promedio_dedos_z - palm.z) < 0.05

    return horizontal and no_frontal

# =========================
# PUNTO DE AGARRE
# =========================

def obtener_punto_agarre(hand_landmarks, frame_shape, gesto):
    h, w, _ = frame_shape

    palm = hand_landmarks.landmark[9]
    wrist = hand_landmarks.landmark[0]

    x = int(palm.x * w)
    y = int(palm.y * h)

    if gesto == "OFFER":
        return (x, y - 40)

    elif gesto == "STOP":
        dx = int((palm.x - wrist.x) * w * 0.5)
        dy = int((palm.y - wrist.y) * h * 0.5)
        return (x + dx, y + dy)

    return (x, y)

# =========================
# DETECCIÓN DE OBJETO SIMPLE
# =========================

def detectar_objeto_simple(frame, hand_landmarks):
    h, w, _ = frame.shape

    palm = hand_landmarks.landmark[9]
    cx = int(palm.x * w)
    cy = int(palm.y * h)

    size = 60
    x1 = max(cx - size, 0)
    y1 = max(cy - size, 0)
    x2 = min(cx + size, w)
    y2 = min(cy + size, h)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

    return edge_density > 0.05

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

    if results.multi_hand_landmarks:
        hay_mano = True

        for hand_landmarks in results.multi_hand_landmarks:

            gesto = "HAND"

            # =========================
            # DETECCIÓN DE GESTO
            # =========================
            if es_offer(hand_landmarks):
                gesto = "OFFER"
                color = (0, 200, 255)

            elif es_palm_stop(hand_landmarks):
                gesto = "STOP"
                color = (0, 255, 255)

            # =========================
            # OBJETO + AGARRE
            # =========================
            punto = obtener_punto_agarre(hand_landmarks, frame.shape, gesto)
            hay_objeto = detectar_objeto_simple(frame, hand_landmarks)

            # dibujar punto
            cv2.circle(frame, punto, 8, (0, 0, 255), -1)

            if hay_objeto:
                cv2.putText(frame, "OBJECT",
                            (punto[0], punto[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

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

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()