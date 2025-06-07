import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import math
import tempfile

# Configuração do MediaPipe
mp_holistic = mp.solutions.holistic

# Funções de cálculo de ângulo e RULA
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def calculate_rula_score_tronco(angle):
    if angle <= 5:
        return 1
    elif angle <= 20:
        return 2
    elif angle <= 60:
        return 3
    return 4

def calculate_rula_score_pescoco(angle):
    if angle <= 10:
        return 1
    elif angle <= 20:
        return 2
    return 3

def calculate_rula_score_antebraco(angle):
    if 60 < angle < 100:
        return 1
    return 2

def calculate_rula_score_braco(angle):
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    return 4

# Interface
st.title("📸 Avaliação RULA por Imagem (Foto)")

uploaded_image = st.file_uploader("Escolha uma imagem (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = image.shape

            # Visibilidade ombros
            right_visibility = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].visibility
            left_visibility = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].visibility

            if right_visibility > left_visibility:
                shoulder = mp_holistic.PoseLandmark.RIGHT_SHOULDER
                hip = mp_holistic.PoseLandmark.RIGHT_HIP
                elbow = mp_holistic.PoseLandmark.RIGHT_ELBOW
                wrist = mp_holistic.PoseLandmark.RIGHT_WRIST
                ear = mp_holistic.PoseLandmark.RIGHT_EAR
                side = "Direito"
            else:
                shoulder = mp_holistic.PoseLandmark.LEFT_SHOULDER
                hip = mp_holistic.PoseLandmark.LEFT_HIP
                elbow = mp_holistic.PoseLandmark.LEFT_ELBOW
                wrist = mp_holistic.PoseLandmark.LEFT_WRIST
                ear = mp_holistic.PoseLandmark.LEFT_EAR
                side = "Esquerdo"

            # Coordenadas
            shoulder_coord = (int(landmarks[shoulder].x * width), int(landmarks[shoulder].y * height))
            hip_coord = (int(landmarks[hip].x * width), int(landmarks[hip].y * height))
            elbow_coord = (int(landmarks[elbow].x * width), int(landmarks[elbow].y * height))
            wrist_coord = (int(landmarks[wrist].x * width), int(landmarks[wrist].y * height))
            ear_coord = (int(landmarks[ear].x * width), int(landmarks[ear].y * height))

            # Cálculo dos ângulos
            angle_tronco = calculate_angle(shoulder_coord, hip_coord, [hip_coord[0], hip_coord[1] - 1])
            angle_pescoco = calculate_angle(shoulder_coord, ear_coord, hip_coord)
            angle_antebraco = calculate_angle(shoulder_coord, elbow_coord, wrist_coord)
            angle_braco = calculate_angle(hip_coord, shoulder_coord, wrist_coord)

            # RULA scores
            rula_tronco = calculate_rula_score_tronco(angle_tronco)
            rula_pescoco = calculate_rula_score_pescoco(angle_pescoco)
            rula_antebraco = calculate_rula_score_antebraco(angle_antebraco)
            rula_braco = calculate_rula_score_braco(angle_braco)

            # Exibir na imagem
            cv2.putText(image, f'Tronco {side}: {int(angle_tronco)} (RULA {rula_tronco})', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f'Pescoco {side}: {int(angle_pescoco)} (RULA {rula_pescoco})', (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f'Antebraco {side}: {int(angle_antebraco)} (RULA {rula_antebraco})', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f'Braco {side}: {int(angle_braco)} (RULA {rula_braco})', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Desenho dos pontos e linhas
            cv2.circle(image, shoulder_coord, 5, (255, 0, 0), -1)
            cv2.circle(image, hip_coord, 5, (255, 0, 0), -1)
            cv2.line(image, shoulder_coord, hip_coord, (255, 0, 0), 2)
            cv2.circle(image, elbow_coord, 5, (0, 0, 255), -1)
            cv2.line(image, shoulder_coord, elbow_coord, (0, 0, 255), 2)
            cv2.circle(image, wrist_coord, 5, (0, 255, 255), -1)
            cv2.line(image, elbow_coord, wrist_coord, (0, 255, 255), 2)
            cv2.circle(image, ear_coord, 5, (0, 255, 0), -1)
            cv2.line(image, shoulder_coord, ear_coord, (0, 255, 0), 2)

            st.image(image, channels="BGR", caption="Imagem com análise RULA")

            # Download
            _, img_encoded = cv2.imencode(".png", image)
            st.download_button(
                label="📥 Baixar imagem processada",
                data=img_encoded.tobytes(),
                file_name="imagem_rula_processada.png",
                mime="image/png"
            )
        else:
            st.warning("Não foram detectados pontos de pose na imagem.")
