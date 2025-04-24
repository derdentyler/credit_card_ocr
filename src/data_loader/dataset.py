import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List

class CreditCardDataset(Dataset):
    """
    Dataset для детекции номера карты.
    - real: множество боксов, выбираем 4-е вхождение class==2.
    - synthetic: ровно один бокс, сохраняем его.
    Структура базовой папки:
      base_dir/
        images/{train,val,test}/
        labels/{train,val,test}/
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        img_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
        transforms: Optional[callable] = None,
        target_class: int = 2,
        target_idx: int = 3
    ):
        super().__init__()
        base = Path(base_dir)
        self.img_dir = base / 'images' / split
        self.lbl_dir = base / 'labels' / split
        self.transforms = transforms
        self.target_class = target_class
        self.target_idx = target_idx

        # Сканируем пары (картинка, аннотация)
        self.samples: List[Tuple[Path, Path]] = []
        for p in self.img_dir.iterdir():
            if p.suffix.lower() in img_exts:
                lbl = self.lbl_dir / (p.stem + '.txt')
                if lbl.exists():
                    self.samples.append((p, lbl))
        if not self.samples:
            raise RuntimeError(f"No images in {self.img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # Читаем все строки с нужным классом
        lines = lbl_path.read_text().splitlines()
        cls2_lines = [l for l in lines if int(l.split()[0]) == self.target_class]

        # Выбираем либо 4-е вхождение (idx=3), либо первое, если оно одно
        if len(cls2_lines) > self.target_idx:
            chosen = cls2_lines[self.target_idx]
        elif len(cls2_lines) == 1:
            chosen = cls2_lines[0]
        else:
            # Нет подходящих боксов — возвращаем пустой таргет
            return self._empty_sample(img)

        # Парсим выбранный бокс
        _, xc, yc, bw, bh = map(float, chosen.split())
        xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
        x1, y1 = xc - bw/2, yc - bh/2
        x2, y2 = xc + bw/2, yc + bh/2

        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.zeros((1,), dtype=torch.int64)  # единственный класс

        sample = {'image': img, 'boxes': boxes, 'labels': labels}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _empty_sample(self, img: Image.Image) -> dict:
        """Если не найден ни один бокс — возвращаем пустой таргет."""
        return {'image': img,
                'boxes': torch.zeros((0,4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64)}
