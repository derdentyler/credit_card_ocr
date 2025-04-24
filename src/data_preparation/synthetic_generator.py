import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from typing import Optional, Tuple, List
from src.utils.logger import logger

class SyntheticCardGenerator:
    def __init__(
        self,
        output_dir: str,
        font_paths: List[str],
        background_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (800, 800),
        card_scale: float = 0.3,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_template: str = "#### #### #### ####",
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.font_paths = [Path(p) for p in font_paths]
        self.background_dir = Path(background_dir) if background_dir else None
        self.image_size = image_size
        self.card_scale = card_scale
        self.bg_color = bg_color
        self.text_template = text_template

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        if self.background_dir and self.background_dir.exists():
            self.background_paths = list(self.background_dir.glob("*.*"))
        else:
            self.background_paths = []

    def _random_number(self) -> str:
        return ''.join(str(random.randint(0, 9)) if ch == '#' else ch for ch in self.text_template)

    def _load_background(self) -> Image.Image:
        if self.background_paths:
            path = random.choice(self.background_paths)
            bg = Image.open(path).convert("RGB")
            return bg.resize(self.image_size)
        return Image.new("RGB", self.image_size, self.bg_color)

    def _draw_card(self, number: str) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        canvas = self._load_background().convert("RGBA")
        w_img, h_img = self.image_size

        # Размеры карты
        card_h = int(h_img * self.card_scale)
        aspect = 85.6 / 53.98
        card_w = int(card_h * aspect)

        # Случайная позиция карты в пределах кадра
        x0 = random.randint(0, w_img - card_w)
        y0 = random.randint(0, h_img - card_h)
        x1, y1 = x0 + card_w, y0 + card_h

        # --- Drop shadow ---
        shadow_offset = (5, 5)
        shadow_radius = 8
        shadow_layer = Image.new("RGBA", self.image_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_rect = [x0 + shadow_offset[0], y0 + shadow_offset[1], x1 + shadow_offset[0], y1 + shadow_offset[1]]
        shadow_draw.rounded_rectangle(shadow_rect, radius=int(min(card_w, card_h)*0.06), fill=(0, 0, 0, 120))
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_radius))  # размытие тени :contentReference[oaicite:1]{index=1}

        # Помещаем тень на канвас
        canvas = Image.alpha_composite(canvas, shadow_layer)

        # --- Карта ---
        draw = ImageDraw.Draw(canvas)
        fill = tuple(random.randint(200, 255) for _ in range(3))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=int(min(card_w, card_h)*0.06), fill=fill)

        # --- Номер карты ---
        # ищем максимальный размер шрифта, чтобы текст занимал ≤80% ширины и ≤10% высоты карты
        best_font = None
        best_tw = best_th = 0
        for size in range(int(card_h*0.1), int(card_h*0.05), -1):
            font = ImageFont.truetype(str(random.choice(self.font_paths)), size)
            bbox = font.getbbox(number)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= card_w * 0.8 and th <= card_h * 0.1:
                best_font, best_tw, best_th = font, tw, th
                break
        if not best_font:
            best_font = ImageFont.truetype(str(random.choice(self.font_paths)), int(card_h*0.05))
            bbox = best_font.getbbox(number)
            best_tw = bbox[2] - bbox[0]
            best_th = bbox[3] - bbox[1]

        # Случайный небольшой вертикальный сдвиг текста (±10% высоты карты)
        v_shift = random.randint(-int(card_h*0.1), int(card_h*0.1))
        tx = x0 + (card_w - best_tw)//2
        ty = y0 + (card_h - best_th)//2 + v_shift

        draw.text((tx, ty), number, font=best_font, fill=(0, 0, 0))

        return canvas.convert("RGB"), (x0, y0, x1, y1)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        if random.random() < 0.2:
            arr = np.array(img).astype(np.float64)
            noise = np.random.normal(0, 5, arr.shape)
            arr += noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img

    def generate(self, count: int):
        logger.info(f"Generating {count} synthetic images...")
        w_img, h_img = self.image_size
        for i in range(count):
            number = self._random_number()
            img, (x0, y0, x1, y1) = self._draw_card(number)
            img = self._augment(img)

            name = f"syn_{i:05d}"
            img.save(self.images_dir / f"{name}.jpg", quality=95)

            # Аннотация YOLO: class, x_center, y_center, width, height
            xc = ((x0 + x1) / 2) / w_img
            yc = ((y0 + y1) / 2) / h_img
            bw = (x1 - x0) / w_img
            bh = (y1 - y0) / h_img
            with open(self.labels_dir / f"{name}.txt", 'w') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        logger.info("Synthetic generation complete.")
