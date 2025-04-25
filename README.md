# Credit Card OCR

## Description
This project implements an end-to-end pipeline for detecting and reading 16-digit credit card numbers from images. It uses YOLOX-S for region detection and EasyOCR for digit recognition. Both real (CreditCardDetectionDS) and synthetic data are supported.

## Architecture

### Modules

- **src/data_loader/**
  - `dataset.py` – PyTorch `Dataset` that loads images and YOLO-format labels, filters only the credit-number box.
  - `transforms.py` – Simple image transforms (e.g. horizontal flip).

- **src/ocr/**
  - `digit_recognizer.py` – OCR module (EasyOCR) for reading cropped number regions.

- **src/data_preparation/**
  - `synthetic_generator.py` – Script to generate synthetic credit-card images with bounding-box annotations.

- **YOLOX/exps/creditcard/**
  - `yolox_cc_s.py` – YOLOX “Exp” file defining model architecture, data loaders (using `CreditCardDataset`), and training schedule.

- **YOLOX/data/**
  - `creditcard.yaml` – Config for real data.
  - `creditcard_synth.yaml` – Config for synthetic-data-only training.
  - `creditcard_mix.yaml` – Config for mixed synthetic + real data training.

- **scripts/**
  - `visualize_boxes.py` – Tool to overlay YOLO-format boxes on images.

## What Has Been Done
- **Detection system** using YOLOX-S, fully integrated via custom `yolox_cc_s.py`.
- **Recognition system** using EasyOCR (`digit_recognizer.py`) for extracted regions.
- **Data loaders** (`CreditCardDataset`) handle real and synthetic annotations, filtering only the card-number box.
- **Synthetic data generation** with `synthetic_generator.py`: card positioning, rounded corners, shadows, font sizing, and basic augmentations.
- **Logging** implemented with TensorBoard and optional MLflow (see below).

## What Has Not Been Done
- Full multi-stage training: due to time constraints, model has not been trained end-to-end.  
  Intended two-stage schedule:
  1. Pre-train on synthetic data (`YOLOX/data/creditcard_synth.yaml`).
  2. Fine-tune on synthetic + real data (`YOLOX/data/creditcard_mix.yaml`).
- Experiment tracking server (MLflow) not yet fully set up; integration steps provided below.

## Installation

1. **Create & activate Python environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Editable install YOLOX** (to make `import yolox` work)  
   ```bash
   pip install -e YOLOX
   ```

4. **Clone dataset**  
   ```bash
   mkdir -p datasets
   cd datasets
   git lfs install
   git clone https://huggingface.co/datasets/bytesWright/creditCardDetectionDS
   cd ..
   ```

5. **Download YOLOX weights**  
   ```bash
   wget -P YOLOX/weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1/yolox_s.pth
   ```

6. **Assets**  
   - Create `assets/fonts/` and add `ocra.ttf`, `CreditCard.ttf`.  
   - Create `assets/backgrounds/` and add sample backgrounds (wood.jpg, stone.jpg, table.jpg).

## Current Train
### Only original dataset creditCardDetectionDS
```cmd
set CUDA_VISIBLE_DEVICES=-1
python YOLOX/tools/train.py ^
  -f YOLOX/exps/creditcard/yolox_cc_s.py ^
  -d -1 -b 8 --logger tensorboard ^
  --opts data_dir=YOLOX/data/creditcard.yaml max_epoch=20
```

## Best way Train (to do)
### Stage 1: Synthetic-only
```cmd
set CUDA_VISIBLE_DEVICES=-1
python YOLOX/tools/train.py ^
  -f YOLOX/exps/creditcard/yolox_cc_s.py ^
  -d -1 -b 8 --logger tensorboard ^
  --opts data_dir=YOLOX/data/creditcard_synth.yaml max_epoch=20
```

### Stage 2: Synthetic + Real
```cmd
set CUDA_VISIBLE_DEVICES=-1
python YOLOX/tools/train.py ^
  -f YOLOX/exps/creditcard/yolox_cc_s.py ^
  -d -1 -b 8 --logger tensorboard ^
  --opts data_dir=YOLOX/data/creditcard_mix.yaml max_epoch=50
```

## OCR Demo (after model is trained)
```bash
python infer.py   --exp-file YOLOX/exps/creditcard/yolox_cc_s.py   --weights YOLOX/weights/yolox_s.pth   --image-path path/to/card.jpg   --output-path out/demo.jpg   --conf 0.3
```

## Serving & Experiments

### TensorBoard
Run during training with `--logger tensorboard`; view with:
```bash
tensorboard --logdir runs
```

### MLflow integration
1. **Install** (added to `requirements.txt`):  
   ```bash
   pip install mlflow
   ```
2. **Start tracking server**:
   ```bash
   mlflow ui --backend-store-uri ./mlruns
   ```
3. **Log experiments**: use `--logger mlflow` in training commands.

## Plans
- Small bugfix
- Unit tests
- Provide REST API for real-time inference
- Add Docker support

## Contact
For any questions or suggestions, feel free to reach out: 📧 [alexander.polybinsky@gmail.com]()