import pytest
from pathlib import Path
from src.data_loader.dataset import CreditCardDataset

@pytest.fixture
def sample_dataset():
    """Фикстура для загрузки датасета с целевым классом 2 (номер карты) и третий bbox (target_idx=3)."""
    return CreditCardDataset(
        base_dir="datasets/creditCardDetectionDS"
    )

def test_dataset_loads(sample_dataset):
    """Проверяет, что датасет загружается и содержит данные."""
    assert len(sample_dataset) > 0, "Датасет пуст!"
    sample = sample_dataset[0]
    assert "image" in sample, "Изображение не найдено в образце!"
    assert "boxes" in sample, "Bbox не найдены в образце!"
    assert sample["boxes"].shape[1] == 4, "Неверная форма bbox (ожидается [x_center, y_center, width, height])"

def test_bbox_normalization(sample_dataset):
    """Проверяет, что координаты bbox нормализованы (в диапазоне [0, 1])."""
    sample = sample_dataset[0]
    x_center, y_center, width, height = sample["boxes"][0]
    assert 0 <= x_center <= 1, f"x_center={x_center} не нормализован!"
    assert 0 <= y_center <= 1, f"y_center={y_center} не нормализован!"
    assert 0 < width <= 1, f"width={width} не нормализован!"
    assert 0 < height <= 1, f"height={height} не нормализован!"
