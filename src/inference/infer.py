#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from YOLOX.yolox.utils import load_ckpt
from YOLOX.yolox.exp import get_exp
from src.ocr.digit_recognizer import DigitRecognizer

def parse_boxes(detections: torch.Tensor, conf_threshold: float = 0.3):
    """
    Преобразует выход модели YOLOX в список координат боксов.
    Ожидается, что detections — тензор формы [1, N, 6], где
    последние два значения в каждой строке — (score, class_id).
    Отбрасываем все предсказания с confidence < conf_threshold.
    Возвращаем список кортежей (x1, y1, x2, y2).
    """
    # берем первую (и единственную) запись из батча
    det = detections[0].detach().cpu().numpy()
    boxes = []
    for *xyxy, score, cls in det:
        if score < conf_threshold:
            continue
        # приводим координаты к целым
        x1, y1, x2, y2 = map(int, xyxy)
        boxes.append((x1, y1, x2, y2))
    return boxes

def load_model(exp_file: str, weights_path: str, device: torch.device):
    """
    Загружает YOLOX-модель по описанию Exp и весам.
    - exp_file: путь к файлу exp (например, exps/creditcard/yolox_cc_s.py)
    - weights_path: путь к .pth-файлу с весами
    - device: torch.device('cpu') или 'cuda'
    Возвращает готовую к инференсу модель.
    """
    # создаём эксперимент и модель
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    # загружаем веса
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model = load_ckpt(model, state_dict)
    model.to(device).eval()
    return model

def infer_image(model, image_path: Path, conf_threshold: float):
    """
    Выполняет инференс одной картинки:
    1) Загружаем изображение через PIL.
    2) Превращаем в тензор [1,3,H,W], нормируем значения в [0,1].
    3) Прогоняем через модель.
    4) Парсим боксы функцией parse_boxes.
    5) Для каждого бокса обрезаем регион и применяем EasyOCR.
    6) Рисуем рамку и распознанный номер на исходном изображении.
    Возвращает:
      img: PIL.Image с визуализацией
      results: список строк (распознанных номеров)
    """
    # 1) загрузка
    img = Image.open(image_path).convert('RGB')
    # 2) подготовка тензора
    array = np.array(img)
    inp = torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float() / 255.0
    # 3) детекция
    with torch.no_grad():
        dets = model(inp.to(next(model.parameters()).device))
    # 4) распаковка боксов
    boxes = parse_boxes(dets, conf_threshold)
    # 5) OCR
    recognizer = DigitRecognizer()
    results = []
    for (x1,y1,x2,y2) in boxes:
        crop = img.crop((x1, y1, x2, y2))
        number = recognizer.recognize(crop)
        results.append(number)
    # 6) визуализация
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for (x1,y1,x2,y2), num in zip(boxes, results):
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, y1-10), num, fill="red", font=font)
    return img, results

def main():
    parser = argparse.ArgumentParser(
        description="Детекция номера карты YOLOX + распознавание EasyOCR"
    )
    parser.add_argument(
        "--exp-file", "-e", type=str,
        default="YOLOX/exps/creditcard/yolox_cc_s.py",
        help="Путь к файлу Exp для YOLOX"
    )
    parser.add_argument(
        "--weights", "-w", type=str,
        default="YOLOX/weights/yolox_s.pth",
        help="Путь к файлу с весами .pth"
    )
    parser.add_argument(
        "--image-path", "-i", type=str, required=True,
        help="Входное изображение с картой"
    )
    parser.add_argument(
        "--output-path", "-o", type=str,
        help="Куда сохранить результат визуализации (необязательно)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Порог confidence для отбора детекций"
    )
    args = parser.parse_args()

    # определяем устройство: GPU если доступно, иначе CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # загружаем модель
    model = load_model(args.exp_file, args.weights, device)

    # инференс и визуализация
    img, results = infer_image(model, Path(args.image_path), args.conf)
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        print(f"Сохранено: {out}")
    print("Распознанные номера:", results)

if __name__ == "__main__":
    main()
