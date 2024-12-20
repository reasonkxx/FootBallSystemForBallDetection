import os
from ultralytics import YOLO

def train_and_test_yolov8(data_yaml_path, epochs, custom_save_dir="D:/university/KursovaWork/FootBallSystemForBallDetection/experiments/exp1/results"):
    """
    Навчає попередньо натреновану модель YOLOv8 на користувацьких даних і зберігає результати тестування.

    Args:
        data_yaml_path (str): Шлях до YAML-файлу з набором даних.
        epochs (int): Кількість епох для перенавчання моделі.
        custom_save_dir (str): Папка для збереження результатів тестування.
    """
    print("=== Навчання та тестування моделі YOLOv8 ===")
    
    # Завантаження базової моеделі YOLOv8
    model = YOLO('yolov8m.pt')
    
    # Перенавчання моделі на користувацькому наборі даних
    model.train(data=data_yaml_path, epochs=epochs, imgsz=1024, batch=8, device=0)  
     
    # Виконуємо тестування на тестовому наборі даних
    results = model.val(data=data_yaml_path, save=True, save_json=True)

    # Тестування моделі на тестовому наборі даних
    test_results = model.predict(
        data=data_yaml_path,
        save=True,
        save_txt=True,
        save_conf=True,
        batch=4  # Зменшений розмір батчу для тестування
    )

    # Збереження навченої моделі
    print("=== Процес збереження моделі ===")
    export_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/models/trained_model/"
    os.makedirs(export_path, exist_ok=True)
    
    # # Экспорт модели в формате torchscript
    # model.export(format="torchscript", dynamic=True, imgsz=640, project=export_path, name="yolov8n_trained")
    # print(f"Модель збережена за шляхом: {export_path}")
    # print(f"Результати тестування збережено в папку: {custom_save_dir}")
    # print("=== Навчання і тестування завершено ===") 
    # return results, test_results

if __name__ == "__main__":
    # Шлях до конфігураційного файлу YAML
    data_yaml_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/dataset.yaml"
    
    # Запуск перенавчання і тестування моделі YOLOv8
    train_and_test_yolov8(data_yaml_path, epochs=200)
