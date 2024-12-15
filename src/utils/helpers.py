import os
from ultralytics import YOLO

def test_pretrained_yolov8(source_path, custom_save_dir="D:/university/KursovaWork/FootBallSystemForBallDetection/experiments/pretrained_results"):
    """
    Тестування попередньо натренованої моделі YOLOv8 без додаткового навчання на користувацьких даних.

    Args:
        source_path (str): Шлях до тестових даних (зображення, відео або папка з ними).
        custom_save_dir (str): Папка для збереження результатів тестування.
    """
    print("=== Тестування попередньо натренованої моделі YOLOv8 ===")

    # Завантаження попередньо натренованої моделі YOLOv8
    model = YOLO('yolov8n.pt')
    print("Попередньо натренована модель завантажена.")

    # Тестування моделі на тестовому наборі даних
    print("Виконання передбачень на тестовому наборі...")
    results = model.predict(
        source=source_path,  # Вказуємо шлях до тестових даних
        save=True,           # Зберігаємо результати
        save_txt=True,       # Зберігаємо координати рамок у текстовому форматі
        save_conf=True,      # Зберігаємо рівень впевненості для кожного передбачення
        project=custom_save_dir,
        name="pretrained_yolo_results",
        batch=4,             # Розмір пакету для швидшого тестування
        device='cpu'         # Використання CPU
    )

    print("=== Тестування завершено ===")
    print(f"Результати тестування збережено в папку: {custom_save_dir}")
    return results

if __name__ == "__main__":
    # Шлях до тестових даних (зображення, відео або папка з ними)
    source_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/test_images/"

    # Запуск тестування попередньо натренованої моделі YOLOv8
    test_pretrained_yolov8(source_path)
