import cv2
from ultralytics import YOLO

def detect_ball_in_video(model_path, video_path, output_video_path):
    # Завантаження натренованої моделі
    model = YOLO(model_path)
    
    # Відкриваємо відео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не вдалося відкрити відео: {video_path}")
        return

    # Отримуємо параметри відео (розміри кадру та FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Визначаємо кодек і створюємо VideoWriter для збереження вихідного відео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для запису у форматі MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Проходимо по кадрах відео
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Виконуємо детекцію на поточному кадрі
        results = model.predict(source=frame, save=False, conf=0.25)  # Встановлюємо поріг впевненості
        
        # Отримуємо результати детекції
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координати рамки
                conf = box.conf[0]  # Впевненість детекції
                label = result.names[int(box.cls[0])]  # Ім'я класу

                # Відображаємо рамку і текст на кадрі
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелена рамка
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Зберігаємо оброблений кадр у вихідне відео
        out.write(frame)

        # Відображаємо поточний кадр (необов'язково, для наочності)
        cv2.imshow('Відео з детекцією', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Звільняємо ресурси
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Шлях до натренованої моделі, вхідного відео та вихідного відео
model_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/models/trained/yolov8n_trained.pt"
video_path = "path/to/your/input_video.mp4"  # Замініть на шлях до вашого відео
output_video_path = "path/to/your/output_video.mp4"  # Замініть на шлях для збереження вихідного відео

# Запуск детекції
detect_ball_in_video(model_path, video_path, output_video_path)
