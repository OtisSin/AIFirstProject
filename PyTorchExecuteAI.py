import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from PIL import Image

# Определяем архитектуру нейросети (должна совпадать с архитектурой при обучении)
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# Функция для предсказания цифры на изображении
def predict_digit(image_path, model_path, img_size=64):
    """
    Предсказывает цифру на изображении

    Args:
        image_path: путь к изображению
        model_path: путь к обученной модели
        img_size: размер, к которому будет приведено изображение
    """

    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    # Загружаем модель
    model = DigitCNN()

    try:
        # Пытаемся загрузить модель (несколько вариантов на случай разных форматов сохранения)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Модель успешно загружена!")

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

    model.to(device)
    model.eval()  # Переводим модель в режим оценки

    # Трансформации для предобработки изображения
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Изменяем размер
        transforms.Grayscale(num_output_channels=1),  # Конвертируем в grayscale
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize((0.5,), (0.5,))  # Нормализуем
    ])

    try:
        # Загружаем и обрабатываем изображение
        image = Image.open(image_path)
        print(f"Исходное изображение: {image.size}, режим: {image.mode}")

        # Показываем исходное изображение
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Исходное изображение')
        plt.axis('off')

        # Применяем трансформации
        input_tensor = transform(image).unsqueeze(0).to(device)  # Добавляем batch dimension


        # Показываем обработанное изображение
        plt.subplot(1, 3, 2)
        processed_img = input_tensor.squeeze().cpu().numpy()  # Убираем batch dimension и переводим в numpy
        plt.imshow(processed_img, cmap='gray')
        plt.title('После обработки')
        plt.axis('off')

        # Предсказание
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_prob, predicted_class = torch.max(probabilities, 1)

            # Получаем вероятности для всех классов
            all_probs = probabilities.squeeze().cpu().numpy()

        predicted_digit = predicted_class.item()
        confidence = predicted_prob.item()

        # Выводим подробные результаты
        print(f"\n{'=' * 50}")
        print(f"РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:")
        print(f"{'=' * 50}")

        # Показываем топ-3 предсказания
        print("\nТоп-3 предсказания:")
        top_probs, top_indices = torch.topk(probabilities, 3)
        for i, (prob, idx) in enumerate(zip(top_probs.squeeze(), top_indices.squeeze())):
            print(f"{i + 1}. Цифра {idx.item()}: {prob.item():.4f} ({prob.item() * 100:.2f}%)")

        return predicted_digit, confidence

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None


# Основная функция
if __name__ == "__main__":
    # Пути к модели и изображению
    model_path = "digit_cnn_model.pth"  # Убедитесь, что файл существует
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "1.png"  # Ваше изображение с цифрой



    print(" Запуск распознавания цифры...")
    print(f"Модель: {model_path}")
    print(f"Изображение: {image_path}")
    print()

    # Распознаем цифру на изображении
    result = predict_digit(image_path, model_path, img_size=64)

    if result:
        digit, confidence = result
        print(f"\n Распознавание завершено! На изображении цифра: {digit}")
    else:
        print("\n Не удалось распознать цифру на изображении")