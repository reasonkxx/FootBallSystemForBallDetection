import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CustomYoloDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        label_path = os.path.join(self.labels_path, img_name.replace('.jpg', '.txt'))

        # Загрузка изображения
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загрузка аннотаций
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    for line in lines:
                        values = line.strip().split()
                        if len(values) == 5:
                            class_id, x_center, y_center, width, height = map(float, values)
                            boxes.append([class_id, x_center, y_center, width, height])
        boxes = np.array(boxes) if boxes else np.empty((0, 5))

        # Применяем преобразования, если они заданы
        if self.transform:
            image = self.transform(image)

        return image, boxes

def train_test_model(config_path, epochs=5, batch_size=4, learning_rate=0.001):
    # Загружаем конфигурацию из файла YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if config is None:
        raise ValueError("Ошибка: не удалось загрузить конфигурацию из файла YAML. Проверьте путь к файлу и его содержимое.")
    
    # Путь к данным для тренировки, валидации и тестирования
    train_images_path = config.get('train', None)
    val_images_path = config.get('val', None)
    test_images_path = config.get('test', None)
    
    if not train_images_path or not val_images_path or not test_images_path:
        raise ValueError("Ошибка: конфигурационный файл не содержит необходимых путей к данным (train, val, test).")
    
    train_labels_path = os.path.join(os.path.dirname(train_images_path), '../labels/train')
    val_labels_path = os.path.join(os.path.dirname(val_images_path), '../labels/valid')
    test_labels_path = os.path.join(os.path.dirname(test_images_path), '../labels/test')
    
    # Определяем преобразования для изображений
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Загрузка данных
    train_dataset = CustomYoloDataset(train_images_path, train_labels_path, transform=transform)
    val_dataset = CustomYoloDataset(val_images_path, val_labels_path, transform=transform)
    test_dataset = CustomYoloDataset(test_images_path, test_labels_path, transform=transform)
    
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    # Определяем простую нейронную сеть (например, ResNet18)
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, config['nc'])  # Выходной слой с количеством классов
    
    # Определяем устройство для выполнения (GPU или CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Определяем функцию потерь и оптимизатор
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Тренировка модели
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = zip(*data)
            inputs = torch.stack(inputs).to(device)
            labels = torch.tensor([int(boxes[0][0]) if len(boxes) > 0 else 0 for boxes in labels]).to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход, функция потерь, обратный проход и оптимизация
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Выводим статистику потерь
            running_loss += loss.item()
            if i % 100 == 99:    # Каждые 100 мини-партий
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Обучение завершено')

    # Тестирование модели на валидационном наборе данных
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images, labels = zip(*data)
            images = torch.stack(images).to(device)
            labels = torch.tensor([int(boxes[0][0]) if len(boxes) > 0 else 0 for boxes in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total > 0:
        print(f'Точность на валидационном наборе: {100 * correct / total:.2f}%')
    else:
        print('Нет данных для оценки точности на валидационном наборе.')

if __name__ == "__main__":
    # Путь к конфигурационному файлу YAML
    config_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/dataset.yaml"
    
    # Запускаем обучение и тестирование модели
    train_test_model(config_path, epochs=5, batch_size=8, learning_rate=0.001)
