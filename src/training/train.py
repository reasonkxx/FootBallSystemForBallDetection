import os
import shutil
import yaml
import glob
from ultralytics import YOLO

def train_and_test_yolov8(data_yaml_path, epochs=10, custom_save_dir="D:/university/KursovaWork/FootBallSystemForBallDetection/experiments/exp1/results"):
    """
    Тренирует предобученную модель YOLOv8 на пользовательских данных и сохраняет результаты тестирования.

    Args:
        data_yaml_path (str): Путь к YAML-файлу набора данных.
        epochs (int): Количество эпох для дообучения модели.
        custom_save_dir (str): Папка для перемещения результатов тестирования.
    """
    print("=== Тренировка и тестирование модели YOLOv8 ===")
    
    # Загрузка предобученной модели YOLOv8
    model = YOLO('yolov8n.pt')
    
    # Дообучение модели на пользовательском наборе данных
    model.train(data=data_yaml_path, epochs=epochs, imgsz=416, batch=8, device=0)  # Уменьшенный размер изображения и батч  # Использование GPU
    
    # Выполняем тестирование на тестовом наборе данных и сохраняем результаты в указанную папку
    results = model.val(data=data_yaml_path, save=True, save_json=True)

    # Тестирование модели на тестовом наборе данных
    test_results = model.predict(data=data_yaml_path, save=True, save_txt=True, save_conf=True, save_dir=custom_save_dir, batch=4)  # Уменьшенный размер батч для тестирования

    
    
    # Сохранение обученной модели в папку 'trained'
    print("=== Процесс сохранения модели ===")
    trained_model_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/models/trained/yolov8n_trained.pt"
    os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
    model.save(trained_model_path)
    print(f"Модель сохранена по пути: {trained_model_path}")
    print(f"Результаты тестирования сохранены в папку: {custom_save_dir}")
    print("=== Тренировка и тестирование завершены ===")
    return results, test_results

if __name__ == "__main__":
    # Путь к конфигурационному файлу YAML
    data_yaml_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/dataset.yaml"
    
    # Запуск дообучения и тестирования модели YOLOv8
    train_and_test_yolov8(data_yaml_path, epochs=2)
