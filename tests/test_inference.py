import pytest
import torch
from src.inference.infer import load_model

def test_model_initialization():
    model = load_model(config_path="src/train/config.yaml")
    dummy_input = torch.randn(1, 3, 800, 800)
    output = model(dummy_input)
    assert output.shape[1] == 4  # [x_center, y_center, width, height]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_inference_speed():
    model = load_model()
    start_time = time.time()
    model.predict("test_image.jpg")
    assert time.time() - start_time < 2.0  # < 2 секунд на GPU