import pytest
from src.data_preparation.synthetic_generator import SyntheticCardGenerator
from PIL import Image


@pytest.fixture
def generator():
    return SyntheticCardGenerator(
        output_dir="tmp/synthetic_test",
        font_paths=["assets/fonts/OCRA.ttf"],
        image_size=(800, 800)
    )


def test_synthetic_generation(generator):
    generator.generate(5)

    images = list(generator.images_dir.glob("*.jpg"))
    labels = list(generator.labels_dir.glob("*.txt"))

    assert len(images) == 5
    assert len(labels) == 5


def test_synthetic_image_size(generator):
    img, _ = generator._draw_card("4111 1111 1111 1111")
    assert img.size == (800, 800)