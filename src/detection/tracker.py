from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 



class BallTracker:
    def __init__(self, max_disappeared=10):
        self.last_position = None
        self.disappeared = 0
        self.max_disappeared = max_disappeared

    def update(self, detections):
        # Проверяем, найдены ли объекты
        ball_detected = False
        for detection in detections:
            for box in detection.boxes:
                label = detection.names[int(box.cls[0])]
                if label == "ball":
                    self.last_position = list(map(int, box.xyxy[0]))
                    self.disappeared = 0
                    ball_detected = True
                    return self.last_position
    
    

        # Если мяча не обнаружено, увеличиваем счетчик исчезновений
        if not ball_detected:
            self.disappeared += 1
            if self.disappeared <= self.max_disappeared and self.last_position is not None:
                return self.last_position  # Возвращаем последнюю известную позицию
            else:
                self.last_position = None  # Сбрасываем, если мяч долго отсутствует
        return None
        
    def is_referee(frame, x1, y1, x2, y2):
        # Вирізаємо область рамки
        roi = frame[y1:y2, x1:x2]
        # Перетворюємо область у HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Визначаємо діапазон кольорів для форми рефері (наприклад, чорний)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        # Створюємо маску для чорного кольору
        mask = cv2.inRange(hsv_roi, lower_black, upper_black)
        # Розраховуємо відсоток чорного кольору
        black_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
        return black_ratio > 0.5  # Якщо більше 50% чорного, то це рефері
   