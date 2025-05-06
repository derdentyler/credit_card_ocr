# Credit Card OCR

## Description

This project implements an end-to-end pipeline for detecting and reading 16-digit credit card numbers from images. It uses YOLOX for region detection and EasyOCR for digit recognition. Both real (CreditCardDetectionDS) and synthetic data are supported.

## Architecture

### Research
We compared several approaches:
1. **End-to-end OCR** (CRNN on the full card)  
   – Simple but fragile to varied backgrounds.  
2. **Char-by-char detection + assembly**  
   – Fine control per character but complex post-processing.  
3. **YOLOX-S + OCR on whole number region** (chosen)  
   – Robust region detection, standardized OCR input, fast fine-tuning.

## Modules

- **src/data_loader/**  
  - `dataset.py` – loads images + YOLO labels, filters to card-number box.  
  - `transforms.py` – basic image transforms.

- **src/ocr/**  
  - `digit_recognizer.py` – EasyOCR wrapper for reading cropped regions.

- **src/data_preparation/**  
  - `synthetic_generator.py` – generates synthetic card images + YOLO annotations.

- **custom_yolox/data/**  
  - `creditcard.yaml` – real-data config.

- **custom_yolox/exps/creditcard/**  
  - `yolox_cc_s.py` – YOLOX experiment using `CreditCardDataset`.

- **scripts/**  
  - `visualize_boxes.py` – overlay YOLO boxes on images.

## What Has Been Done

- Detection with YOLOX-S via custom `yolox_cc_s.py`.  
- OCR with EasyOCR (`digit_recognizer.py`).  
- Synthetic data generator (`synthetic_generator.py`).  
- Unit tests for synthetic generator and dataset loader.

## What Is Left

- Multi-stage training not run end-to-end yet:  
  1. Synthetic-only (`creditcard_synth.yaml`).  
  2. Synthetic + Real (`creditcard_mix.yaml`).  
- MLflow experiment tracking.

## Installation

1. **Clone project**  
   ```bash
   git clone https://github.com/derdentyler/credit_card_ocr.git
   cd credit_card_ocr
   ```

2. **Create & activate environment**  
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Clone YOLOX & use symlinks**  
   ```bash
   git clone https://github.com/Megvii-BaseDetection/YOLOX.git
   # Windows PowerShell as Admin:
   New-Item -ItemType SymbolicLink -Path YOLOX\data\creditcard.yaml -Target custom_yolox\data\creditcard.yaml
   New-Item -ItemType SymbolicLink -Path YOLOX\exps\creditcard\yolox_cc_s.py -Target custom_yolox\exps\creditcard\yolox_cc_s.py
   ```

5. **Download YOLOX weights**  
   ```bash
   wget -P YOLOX/weights https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1/yolox_s.pth
   ```

6. **Dataset**  
   ```bash
   mkdir -p datasets
   cd datasets
   git lfs install
   git clone https://huggingface.co/datasets/bytesWright/creditCardDetectionDS
   cd ..
   ```

7. **Assets**  
   - Place fonts in `assets/fonts/` (e.g., `ocra.ttf`, `CreditCard.ttf`).  
   - Place backgrounds in `assets/backgrounds/`.

## Tests

Run all tests:
```bash
pytest tests
```

## Training

### Stage 1: Synthetic-only
```cmd
set CUDA_VISIBLE_DEVICES=-1
python YOLOX/tools/train.py -f exps/creditcard/yolox_cc_s.py -d -1 -b 8 --logger tensorboard --opts data_dir=YOLOX/data/creditcard_synth.yaml max_epoch=20
```

### Stage 2: Synthetic + Real
```cmd
set CUDA_VISIBLE_DEVICES=-1
python YOLOX/tools/train.py -f exps/creditcard/yolox_cc_s.py -d -1 -b 8 --logger tensorboard --opts data_dir=YOLOX/data/creditcard_mix.yaml max_epoch=50
```

## OCR Demo

After training:
```bash
python infer.py --exp-file YOLOX/exps/creditcard/yolox_cc_s.py --weights YOLOX/weights/yolox_s.pth --image-path path/to/card.jpg --output-path out/demo.jpg --conf 0.3
```

## Plans

- Run full two-stage training.  
- Integrate MLflow.  
- Add Docker support for reproducible environments.

## Contact
For any questions or suggestions, feel free to reach out: 📧 [alexander.polybinsky@gmail.com]()