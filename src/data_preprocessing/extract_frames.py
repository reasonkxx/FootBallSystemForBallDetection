import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    # Перевіряємо, чи папка для збереження кадрів вже містить файли
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"Кадри вже витягнуті в папку: {output_folder}. Пропускаємо витяг кадрів.")
        return   
     
    # Відкриваємо відеофайл
    video = cv2.VideoCapture(video_path)
    
    # Перевіряємо, чи вдалося відкрити відео
    if not video.isOpened():
        print(f"Не вдалося відкрити відео: {video_path}")
        return

    # Створюємо папку для збереження кадрів, якщо вона не існує
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_id = 0
    success, frame = video.read()
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Отримуємо кількість кадрів в секунду
    frame_interval = fps // frame_rate  # Інтервал кадрів для витягання
    
    while success:
        # Зберігаємо кадр кожен frame_interval кадр
        if frame_id % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_id:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Кадр збережено: {frame_filename}")
        
        success, frame = video.read()
        frame_id += 1
    
    video.release()
    print(f"Витяг кадрів завершено. Загалом збережено {frame_id // frame_interval} кадрів.")

if __name__ == "__main__":
    # Введіть шлях до відео та папку для збереження кадрів
    video_path = "data/raw/videos/videotest1.mp4"
    output_folder = "data/extracted_frames"
    
    # Викликаємо функцію витягування кадрів
    extract_frames(video_path, output_folder, frame_rate=1)  # frame_rate вказує на кількість кадрів, які зберігаються кожну секунду
