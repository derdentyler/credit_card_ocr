import os
from pathlib import Path
from yolox.exp import Exp as MyExp
from yolox.data import TrainTransform, ValTransform
from yolox.evaluators import VOCEvaluator
from torch.utils.data import DataLoader
from src.data_loader.dataset import CreditCardDataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ─── Параметры модели ───────────────────────────────
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (800, 800)
        self.test_size = (800, 800)
        self.max_epoch = 1               # одна эпоха
        self.data_num_workers = 0        # CPU в Windows

        # ─── Простые аугментации ────────────────────────────
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0

        # ─── Имя эксперимента ───────────────────────────────
        self.exp_name = os.path.splitext(os.path.basename(__file__))[0]

        # ─── Корень данных ──────────────────────────────────
        self.base_data_dir = Path(__file__).parents[2] / "datasets" / "creditCardDetectionDS"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        # Train loader
        transform = TrainTransform(
            max_labels=50,
            flip_prob=self.flip_prob,
            hsv_prob=self.hsv_prob
        )
        ds = CreditCardDataset(
            base_dir=str(self.base_data_dir),
            split="train",
            transforms=transform
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.data_num_workers,
            pin_memory=False
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        # Validation loader
        transform = ValTransform(legacy=False)
        ds = CreditCardDataset(
            base_dir=str(self.base_data_dir),
            split="val",
            transforms=transform
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.data_num_workers,
            pin_memory=False
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        return VOCEvaluator(
            dataloader=loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes
        )
