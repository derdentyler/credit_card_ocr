import argparse
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

        card_h = int(h_img * self.card_scale)
        aspect = 85.6 / 53.98  # соотношение сторон карты в мм
        card_w = int(card_h * aspect)

        x0 = random.randint(0, w_img - card_w)
        y0 = random.randint(0, h_img - card_h)
        x1, y1 = x0 + card_w, y0 + card_h

        # тень
        shadow = Image.new("RGBA", self.image_size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        off = 5
        radius = int(min(card_w, card_h) * 0.06)
        sd.rounded_rectangle(
            [x0+off, y0+off, x1+off, y1+off],
            radius=radius,
            fill=(0, 0, 0, 120)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(8))
        canvas = Image.alpha_composite(canvas, shadow)

        # тело карты
        draw = ImageDraw.Draw(canvas)
        fill = tuple(random.randint(200, 255) for _ in range(3))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill)

        # текст номера
        num_w_limit = card_w * 0.8
        num_h_limit = card_h * 0.1
        best_font = None
        for size in range(int(card_h*0.1), int(card_h*0.05), -1):
            font = ImageFont.truetype(str(random.choice(self.font_paths)), size)
            bbox = font.getbbox(number)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            if tw <= num_w_limit and th <= num_h_limit:
                best_font = font
                break
        if best_font is None:
            best_font = ImageFont.truetype(str(random.choice(self.font_paths)), int(card_h*0.05))
            bbox = best_font.getbbox(number)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

        vshift = random.randint(-int(card_h*0.1), int(card_h*0.1))
        tx = x0 + (card_w - tw)//2
        ty = y0 + (card_h - th)//2 + vshift
        draw.text((tx, ty), number, font=best_font, fill=(0, 0, 0))

        return canvas.convert("RGB"), (x0, y0, x1, y1)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        if random.random() < 0.2:
            arr = np.array(img).astype(np.float64)
            arr += np.random.normal(0, 5, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return img

    def generate(self, count: int):
        logger.info(f"Generating {count} synthetic images...")
        w, h = self.image_size
        for i in range(count):
            number = self._random_number()
            img, (x0, y0, x1, y1) = self._draw_card(number)
            img = self._augment(img)
            name = f"syn_{i:05d}"
            img.save(self.images_dir / f"{name}.jpg", quality=95)

            xc = ((x0 + x1) / 2) / w
            yc = ((y0 + y1) / 2) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            with open(self.labels_dir / f"{name}.txt", "w") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\\n")
        logger.info("Synthetic generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic credit card images and YOLO labels"
    )
    parser.add_argument(
        "--num-images", "-n", type=int, required=True,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True,
        help="Directory for generated images and labels"
    )
    parser.add_argument(
        "--fonts", "-f", nargs="+",
        default=["assets/fonts/ocra.ttf", "assets/fonts/CreditCard.ttf"],
        help="List of font file paths"
    )
    parser.add_argument(
        "--background-dir", "-b", type=str, default="assets/backgrounds",
        help="Directory with background images"
    )
    parser.add_argument(
        "--width", type=int, default=800, help="Output image width"
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Output image height"
    )
    parser.add_argument(
        "--scale", type=float, default=0.3,
        help="Card height as fraction of image height"
    )
    args = parser.parse_args()

    gen = SyntheticCardGenerator(
        output_dir=args.output_dir,
        font_paths=args.fonts,
        background_dir=args.background_dir,
        image_size=(args.width, args.height),
        card_scale=args.scale
    )
    gen.generate(args.num_images)
