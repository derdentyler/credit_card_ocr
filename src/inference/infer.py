import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from src.models.yolox_wrapper import YOLOXDetector
from src.ocr.digit_recognizer import DigitRecognizer

def parse_boxes(detections: torch.Tensor, conf_threshold: float = 0.3):
    """
    Превращает выход модели YOLOX в список боксов (x1, y1, x2, y2),
    отбрасывая детекции с confidence < conf_threshold.
    Ожидается, что detections — тензор формы [batch, num_boxes, 6],
    где последние две координаты — (score, class).
    """
    # Берём первый (и единственный) элемент батча
    det = detections[0]
    det = det.detach().cpu().numpy()
    boxes = []
    for *xyxy, score, cls in det:
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        boxes.append((x1, y1, x2, y2))
    return boxes

def infer_image(image_path: str, output_path: str = None):
    # 1) Загрузка изображения
    img = Image.open(image_path).convert('RGB')

    # 2) Детектор
    detector = YOLOXDetector()
    # Преобразуем PIL -> tensor [1, 3, H, W], нормируем 0–1
    array = np.array(img)
    inputs = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 3) Запуск детекции
    detections = detector(inputs)

    # 4) Парсинг боксов
    boxes = parse_boxes(detections, conf_threshold=0.3)

    # 5) OCR по каждому боксу
    recognizer = DigitRecognizer()
    results = []
    for x1, y1, x2, y2 in boxes:
        crop = img.crop((x1, y1, x2, y2))
        number = recognizer.recognize(crop)
        results.append(number)

    # 6) Визуализация и сохранение результата (опционально)
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), num in zip(boxes, results):
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1 - 10), num, fill='red')

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    # 7) Вывод результатов
    print(f"Image: {image_path}")
    print("Detected numbers:", results)
    return results

if __name__ == "__main__":
    # Пример использования:
    # Укажите путь к тестовой картинке и опционально выходной файл
    infer_image("test/cards/card1.jpg", output_path="out/result1.jpg")
