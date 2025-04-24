import numpy as np
from PIL import Image
import easyocr

class DigitRecognizer:
    def __init__(self, languages=None, gpu: bool = False):
        """
        :param languages: список языковых кодов для EasyOCR (например, ['en']).
        :param gpu: True для использования GPU, False — CPU only.
        """
        langs = languages or ['en']
        self.reader = easyocr.Reader(langs, gpu=gpu)  # :contentReference[oaicite:2]{index=2}

    def recognize(self, img: Image.Image) -> str:
        """
        Выполняет OCR на изображении с помощью EasyOCR и возвращает все цифры подряд.
        """
        # EasyOCR принимает numpy array
        arr = np.array(img)
        results = self.reader.readtext(arr, detail=0, allowlist='0123456789')
        # detail=0 возвращает сразу строки без bbox/конфидэнс
        # allowlist ограничивает символы цифрами
        return ''.join(results)
