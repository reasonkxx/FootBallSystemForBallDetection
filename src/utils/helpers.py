import os
from ultralytics import YOLO
import numpy as np

def evaluate_model_on_video(model_path, video_path):
    # Завантаження моделі
    model = YOLO(model_path)
    
    # Виконання предикту на відео
    results = model.predict(video_path, save=True)
    
    # Отримання метрик
    all_confidences = []
    correct_detections = 0
    total_detections = 0

    for result in results:
        # Проходимо по кожному кадру та об'єктам
        for box in result.boxes:
            conf = box.conf[0].cpu().numpy()  # Впевненість детекції
            all_confidences.append(conf)
            total_detections += 1
            
            # Вважаємо правильним, якщо впевненість більше 0.5 (це можна змінити)
            if conf > 0.5:
                correct_detections += 1

    # Розрахунок середньої точності
    mean_confidence = np.mean(all_confidences)
    accuracy = correct_detections / total_detections if total_detections > 0 else 0
    
    print(f"Середня впевненість моделі: {mean_confidence:.2f}")
    print(f"Точність моделі (conf > 0.5): {accuracy:.2%}")
    
    return mean_confidence, accuracy

if __name__ == "__main__":
    # Шляхи до моделей та відео
    base_model_path = 'yolov8n.pt'  # Базова модель YOLOv8
    finetuned_model_path = 'D:/university/KursovaWork/FootBallSystemForBallDetection/runs/detect/train26/weights/best.pt'
    video_path = 'D:/university/KursovaWork/FootBallSystemForBallDetection/data/input_video/test4.mp4'
    
    print("=== Оцінка базової моделі YOLOv8n ===")
    evaluate_model_on_video(base_model_path, video_path)
    
    # print("\n=== Оцінка донавченої моделі YOLOv8 ===")
    # evaluate_model_on_video(finetuned_model_path, video_path)
