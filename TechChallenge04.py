import cv2
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from collections import Counter
import mediapipe as mp
import math

def detect_faces_emotions_activities(video_path, output_video_path, summary_output_path):

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    emotions_list = []
    activities_list = []

    total_frames_analyzed = 0
    anomalies_detected = 0

    frame_number = 0
    previous_activity = None
    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % 30 != 0:
            out.write(frame)
            continue

        total_frames_analyzed += 1

        try:
            results_emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if not isinstance(results_emotion, list):
                results_emotion = [results_emotion]

            for face in results_emotion:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                dominant_emotion = face['dominant_emotion']
                emotions_list.append(dominant_emotion)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except Exception as e:
            pass

        activity, is_anomaly = detect_activity(frame, pose)
        activities_list.append(activity)

        if is_anomaly:
            anomalies_detected += 1

        cv2.putText(frame, activity, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    generate_summary(emotions_list, activities_list, summary_output_path, total_frames_analyzed, anomalies_detected)

def detect_activity(frame, pose_model):
    """
    Detecta atividades no frame usando Mediapipe Pose.

    Parâmetros:
    - frame: O frame atual do vídeo.
    - pose_model: O modelo de pose do Mediapipe.

    Retorna:
    - activity: Uma descrição da atividade detectada.
    - is_anomaly: True se uma anomalia for detectada, False caso contrário.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose_model.process(image_rgb)

    is_anomaly = False  

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_coord(landmark):
            return np.array([landmark.x, landmark.y])

        nose = get_coord(landmarks[mp.solutions.pose.PoseLandmark.NOSE])
        left_shoulder = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER])
        right_shoulder = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
        left_hip = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP])
        right_hip = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
        left_knee = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE])
        right_knee = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE])
        left_ankle = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE])
        right_ankle = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE])
        left_elbow = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW])
        right_elbow = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW])
        left_wrist = get_coord(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST])
        right_wrist = get_coord(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST])

        activity = "Atividade Desconhecida"

        if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
            activity = "Braços Levantados"
        elif is_sitting(left_hip, left_knee, left_ankle) and is_sitting(right_hip, right_knee, right_ankle):
            activity = "Sentado"
        elif is_standing(left_hip, left_knee, left_ankle) and is_standing(right_hip, right_knee, right_ankle):
            activity = "Em Pé"
        else:
            activity = "Atividade Desconhecida"
            is_anomaly = True

    else:
        activity = "Nenhuma pessoa detectada"
        is_anomaly = False

    return activity, is_anomaly

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos.

    Parâmetros:
    - a, b, c: Coordenadas dos pontos.

    Retorna:
    - angle: O ângulo em graus entre os pontos.
    """
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_sitting(hip, knee, ankle):
    """
    Verifica se a perna está dobrada em posição de sentado.

    Parâmetros:
    - hip: Coordenadas do quadril.
    - knee: Coordenadas do joelho.
    - ankle: Coordenadas do tornozelo.

    Retorna:
    - True se estiver sentado, False caso contrário.
    """
    angle = calculate_angle(hip, knee, ankle)
    if angle < 160:
        return True
    else:
        return False

def is_standing(hip, knee, ankle):
    """
    Verifica se a perna está reta em posição de pé.

    Parâmetros:
    - hip: Coordenadas do quadril.
    - knee: Coordenadas do joelho.
    - ankle: Coordenadas do tornozelo.

    Retorna:
    - True se estiver em pé, False caso contrário.
    """
    angle = calculate_angle(hip, knee, ankle)
    if angle >= 160:
        return True
    else:
        return False

def generate_summary(emotions_list, activities_list, summary_output_path, total_frames_analyzed, anomalies_detected):
    """
    Gera um resumo das emoções e atividades detectadas e salva em um arquivo.

    Parâmetros:
    - emotions_list: Lista de emoções detectadas.
    - activities_list: Lista de atividades detectadas.
    - summary_output_path: Caminho para o arquivo de resumo de saída.
    - total_frames_analyzed: Total de frames analisados.
    - anomalies_detected: Número de anomalias detectadas.
    """
    emotion_counts = Counter(emotions_list)
    activity_counts = Counter(activities_list)

    summary_lines = []
    summary_lines.append(f"Total de frames analisados: {total_frames_analyzed}\n")
    summary_lines.append(f"Número de anomalias detectadas: {anomalies_detected}\n\n")

    summary_lines.append("Resumo das Emoções Detectadas:\n")
    if emotions_list:
        for emotion, count in emotion_counts.most_common():
            summary_lines.append(f"- {emotion}: detectada {count} vezes.\n")
    else:
        summary_lines.append("Nenhuma emoção detectada.\n")

    summary_lines.append("\nResumo das Atividades Detectadas:\n")
    if activities_list:
        for activity, count in activity_counts.most_common():
            summary_lines.append(f"- {activity}: detectada {count} vezes.\n")
    else:
        summary_lines.append("Nenhuma atividade detectada.\n")

    print("\n" + "".join(summary_lines))

    with open(summary_output_path, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4') 
summary_output_path = os.path.join(script_dir, 'resumo.txt')

detect_faces_emotions_activities(input_video_path, output_video_path, summary_output_path)