from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def visualize(image_path, label_path, out_path=None):
    """
    Рисует все боксы из label_path на image_path,
    подписывая каждый: [idx:class_id].
    Четвертый (index 3) элемент класса 2 выделяется жёлтым.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Читаем аннотацию
    lines = Path(label_path).read_text().splitlines()
    w, h = img.size

    class_2_indices = [i for i, line in enumerate(lines) if int(line.split()[0]) == 2]
    target_index = class_2_indices[3] if len(class_2_indices) > 2 else None

    for idx, line in enumerate(lines):
        cls, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        # Выделяем нужный бокс жёлтым, остальные красным
        outline_color = "yellow" if idx == target_index else "red"
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=2)

        text = f"{idx}:{int(cls)}"
        draw.text((x1, y1 - 10), text, fill="yellow", font=font)

    if out_path:
        img.save(out_path)
        print(f"Сохранено: {out_path}")
    else:
        img.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualize_boxes.py <image.jpg> <labels.txt> [<out.jpg>]")
        sys.exit(1)
    visualize(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
