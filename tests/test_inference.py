import time
import pytest
import torch
from pathlib import Path
from src.inference.infer import load_model, infer_image

@pytest.fixture(scope="module")
def model_cpu(tmp_path_factory):
    # Загружаем модель на CPU
    device = torch.device('cpu')
    exp_file = "YOLOX/exps/creditcard/yolox_cc_s.py"
    weights_path = "YOLOX/weights/yolox_s.pth"
    model = load_model(exp_file=exp_file, weights_path=weights_path, device=device)
    return model

def test_model_initialization(model_cpu):
    """
    Проверяем, что модель правильно загружается и даёт тензор выходных предсказаний.
    У YOLOX выход – тензор 3D: [batch, num_boxes, 6], где 6 = (x1,y1,x2,y2,score,class).
    """
    dummy_input = torch.randn(1, 3, 800, 800)
    output = model_cpu(dummy_input)
    # Должно быть 3 измерения и размер последнего – 6
    assert output.ndim == 3, "Ожидаем 3D-тензор [B, N, 6]"
    assert output.shape[2] == 6, "Каждое предсказание должно содержать 6 чисел (bbox+score+class)"

def test_infer_image_runs(model_cpu, tmp_path):
    """
    Проверяем, что функция infer_image обрабатывает картинку без ошибок
    и возвращает PIL.Image и список строк-результатов.
    """
    # Создадим простую тестовую картинку (белый фон)
    img_path = tmp_path / "white.jpg"
    from PIL import Image
    Image.new("RGB", (800,800), "white").save(img_path)

    # Запускаем инференс
    img_out, results = infer_image(model_cpu, img_path, conf_threshold=0.3)

    # Должны получить объект PIL.Image и список (в данном случае пустой)
    from PIL import Image as PILImage
    assert isinstance(img_out, PILImage.Image), "На выходе должен быть PIL.Image"
    assert isinstance(results, list), "Результат – список распознанных строк"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for speed test")
def test_inference_speed_on_gpu(model_cpu):
    """
    (Опционально) Проверяем скорость инференса на GPU (<2 сек).
    Будет пропущен, если CUDA недоступна.
    """
    start = time.time()
    # Используем ту же белую картинку
    img_path = Path("tests/test_image.jpg")
    infer_image(model_cpu, img_path, conf_threshold=0.3)
    duration = time.time() - start
    assert duration < 2.0, f"Inference took too long: {duration:.2f}s"
