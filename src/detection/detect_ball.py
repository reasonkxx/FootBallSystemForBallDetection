import cv2
from ultralytics import YOLO
from src.detection.tracker import BallTracker

def load_model(model_path):
    """Завантажує натреновану модель YOLO."""
    model = YOLO(model_path)
    print("Модель завантажена успішно.")
    return model

def open_video(video_path):
    """Відкриває відео для читання."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не вдалося відкрити відео: {video_path}")
    print("Відео успішно відкрите.")
    return cap

def create_video_writer(output_video_path, frame_width, frame_height, fps):
    """Створює об'єкт VideoWriter для запису відео."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        raise ValueError("Ошибка при створенні об'єкта VideoWriter. Перевірте шлях або параметри.")
    print("VideoWriter успішно створено.")
    return out

def detect_objects_in_frame(model, frame):
    """Виконує детекцію об'єктів на кадрі."""
    results = model.predict(source=frame, save=False, conf=0.25)
    return results

def process_detections(frame, results):
    """Обробляє результати детекції та додає рамки та підписи на кадр."""
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if label == "ball":
                print(f"М'яч знайдено з впевненістю {box.conf[0]:.2f}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            # Відображаємо рамку і текст на кадрі
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелена рамка
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def release_resources(cap, out):
    """Звільняє ресурси відео та записувача."""
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Ресурси успішно звільнено.")

def process_frame_with_tracking(model, tracker, frame):
    """Обробляє кадр з використанням трекера."""
    results = detect_objects_in_frame(model, frame)
    position = tracker.update(results)

    # Обробляємо всі знайдені об'єкти
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            
            if label == "player":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif label == "goalkeeper":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            

    # Якщо позиція м'яча відома, додаємо рамку червоного кольору
    if position:
        x1, y1, x2, y2 = position
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


def detect_ball_in_video(model_path, video_path, output_video_path):
    """Головна функція для виконання детекції м'яча на відео."""
    # Завантаження моделі
    model = load_model(model_path)

    # Ініціалізація трекера
    tracker = BallTracker(max_disappeared=5)

    # Відкриваємо відео
    cap = open_video(video_path)

    # Отримуємо параметри відео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Створюємо VideoWriter для запису вихідного відео
    out = create_video_writer(output_video_path, frame_width, frame_height, fps)

    # Проходимо по кадрах відео
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обробляємо кадр з використанням трекера
        frame = process_frame_with_tracking(model, tracker, frame)

        # Зберігаємо оброблений кадр у вихідне відео
        out.write(frame)

        # Відображаємо поточний кадр (необов'язково)
        cv2.imshow('Відео з детекцією', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Звільняємо ресурси
    release_resources(cap, out)

   

