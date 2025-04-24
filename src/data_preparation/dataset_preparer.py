import random
import shutil
import os
from pathlib import Path
from src.utils.logger import logger


class DatasetPreparer:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        move_files: bool = False  # False = копировать (по умолчанию)
    ):
        # Исходная база с subdirs images/ и labels/
        self.root = Path(raw_data_dir)
        self.images_src = self.root / "images"
        self.labels_src = self.root / "labels"

        # Куда складывать split
        self.root_out = Path(output_dir)
        self.images_out = self.root_out / "images"
        self.labels_out = self.root_out / "labels"

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.move_files = move_files

    def prepare(self):
        logger.info("Начинаем подготовку датасета...")
        # 1) Удаляем старые split-папки, если они есть
        for split in ("train", "val", "test"):
            shutil.rmtree(self.images_out / split, ignore_errors=True)
            shutil.rmtree(self.labels_out / split, ignore_errors=True)

        # 2) Создаём пустые папки заново
        for split in ("train", "val", "test"):
            (self.images_out / split).mkdir(parents=True, exist_ok=True)
            (self.labels_out / split).mkdir(parents=True, exist_ok=True)

        # 3) Разбиваем и копируем/перемещаем
        self._split_and_place()
        logger.info("Подготовка завершена.")

    def _split_and_place(self):
        all_imgs = list(self.images_src.glob("*.jpg"))
        random.shuffle(all_imgs)
        n = len(all_imgs)
        n_test = int(n * self.test_ratio)
        n_val  = int(n * self.val_ratio)
        # тестовые первые, затем валидация, остальные — train
        test_imgs  = all_imgs[:n_test]
        val_imgs   = all_imgs[n_test:n_test+n_val]
        train_imgs = all_imgs[n_test+n_val:]

        for imgs, split in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
            for img in imgs:
                lbl = self.labels_src / f"{img.stem}.txt"
                if not lbl.exists():
                    logger.warning(f"Нет разметки для {img.name}, пропускаем.")
                    continue
                # пути назначения
                dst_img = self.images_out / split / img.name
                dst_lbl = self.labels_out / split / lbl.name

                if self.move_files:
                    shutil.move(str(img), str(dst_img))
                    shutil.move(str(lbl), str(dst_lbl))
                else:
                    shutil.copy(str(img), str(dst_img))
                    shutil.copy(str(lbl), str(dst_lbl))
