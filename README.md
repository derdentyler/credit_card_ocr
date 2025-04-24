# Credit Card OCR

## Description
This project implements an end-to-end pipeline for detecting and reading 16-digit credit card numbers from images. It uses YOLOX-S for region detection and EasyOCR for digit recognition. Both real (CreditCardDetectionDS) and synthetic data are supported.

## Architecture

### Modules

- **src/data_loader/**
  - `dataset.py` – PyTorch `Dataset` that loads images and YOLO-format labels, filters only the credit-number box.
  - `transforms.py` – Simple image transforms (e.g. horizontal flip).

- **src/models/**
  - `yolox_wrapper.py` – Wrapper around YOLOX-S for init, forward pass, and loss.
  - `digit_recognizer.py` – OCR module (EasyOCR) for reading cropped number regions.

- **src/train/**
  - `train.py` – Combines real + synthetic data, sets up DataLoader, optimizer, and runs training.
  - `config.yaml` – Hyperparameters, data paths, model settings.

- **YOLOX/exps/creditcard/**
  - `yolox_cc_s.py` – YOLOX “Exp” file pointing to `datasets/creditCardDetectionDS`, defines dataset loaders, augmentations, and training schedule.

- **YOLOX/data/**
  - `creditcard.yaml` – Maps YOLOX to the real data folders and class names.

- **scripts/**
  - `visualize_boxes.py` – Quick tool to overlay YOLO boxes on images.

### Research
We compared three architectures:
1. **End-to-end OCR** (CRNN on full card)  
   – Simple but fragile to backgrounds.
2. **Char-by-char detection + assembly**  
   – Fine control but complex matching.
3. **YOLOX-S + OCR on whole number region** (chosen)  
   – Stable detection, standardized OCR input, fast fine-tuning on real+synthetic.

## Hacks
- **Synthetic data**: random shadows, varied card colors, random angles/positions.
- **Logging**: TensorBoard; per-epoch checkpoints.
- **OCR**: EasyOCR integration in `infer.py`.

## Installation

1. **Install YOLOX**  
   ```bash
   git clone https://github.com/Megvii-BaseDetection/YOLOX.git
   cd YOLOX
   pip install -v -e .
   pip install -r requirements.txt
   ```

2. **Install this project**  
   ```bash
   cd <project_root>
   pip install -r requirements.txt
   ```

3. **Datasets**  
   - Place `CreditCardDetectionDS` under `datasets/creditCardDetectionDS`.
   - Ensure `YOLOX/data/creditcard.yaml` paths match.

4. **Weights**  
   ```bash
   wget -P YOLOX/weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1/yolox_s.pth
   ```

## Run

From the `YOLOX/` directory on Windows (CPU only):
```cmd
set CUDA_VISIBLE_DEVICES=-1 && python tools/train.py -f exps/creditcard/yolox_cc_s.py -d 1 -b 4 --device cpu
```

Or using custom script:
```bash
python src/train/train.py
```

## Serving & Experiments
- Logs to TensorBoard.
- Future: add MLflow/W&B for experiment tracking.

## Plans
- Add MLflow logging & model registry  
- Experiment with lightweight YOLOX-Nano for faster inference  
- Improve OCR by fine-tuning on synthetic distortions  
- Automate synthetic/real data mixing schedules  
- Add a REST API for real-time inference
